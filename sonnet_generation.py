'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import os
import time
import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW
from evaluation import test_sonnet

TQDM_DISABLE = False

# 로그 기록용 함수.
def write_log(message, log_path):
  print(message)

  if log_path is not None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
      f.write(message + "\n")

# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # Last-k layer fine-tuning (참고: arXiv:1911.03090).
    # last_k > 0 이면 전체를 freeze하고 마지막 k개의 트랜스포머 블록만 학습한다.
    # last_k == 0 이면 기존처럼 전체 모델을 fine-tuning 한다.
    last_k = getattr(args, 'last_k', 0)
    if last_k and last_k > 0:
      for param in self.gpt.parameters():
        param.requires_grad = False

      total_layers = len(self.gpt.gpt_layers)
      k = min(last_k, total_layers)
      for layer in self.gpt.gpt_layers[total_layers - k:]:
        for param in layer.parameters():
          param.requires_grad = True

      # 마지막 LayerNorm도 함께 학습 (파라미터 수는 미미하지만 출력 분포 보정에 도움).
      for param in self.gpt.final_layer_norm.parameters():
        param.requires_grad = True
    else:
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
    ### 완성시켜야 할 빈 코드 블록
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)

    # outputs["last_hidden_state"]: [batch_size, seq_len, hidden_size]
    hidden_states = outputs["last_hidden_state"]

    # logits: [batch_size, seq_len, vocab_size]
    logits = self.gpt.hidden_state_to_token(hidden_states)

    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    top-p sampling 과 softmax temperature를 사용하여 새로운 소넷을 생성한다.

    TODO: 지금 이 방법은 기대 이하일 수 있다. 영감을 얻기 위해 Hugging Face의 model.generate(...) 함수를 참고해도 좋겠다.
        여러 시퀀스를 생성하고 beam search를 통해 최적의 시퀀스를 선택하는 것도 좋은 한 가지 방법이다.
        Top-k 샘플링 역시 또 다른 방법이며, 그 외에도 많은 접근법이 있다.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # logits을 구하기 위한 forward pass.
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")
  write_log(f"save the model to {filepath}", args.log_path)

@torch.no_grad()
def compute_dev_loss(model, dev_path, device, batch_size=4):
  """Dev gold 소넷(전체 14줄)에 대한 teacher-forced cross-entropy 평균.
  test 소넷은 쓰지 않는다 -- dev_path 에 dev gold 경로를 넘긴다."""
  model.eval()
  dev_dataset = SonnetsDataset(dev_path)
  dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size,
                          collate_fn=dev_dataset.collate_fn)
  total_loss, num_batches = 0.0, 0
  for batch in dev_loader:
    b_ids = batch['token_ids'].to(device)
    b_mask = batch['attention_mask'].to(device)
    logits = model(b_ids, b_mask)
    logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
    labels = b_ids[:, 1:].contiguous().flatten()
    loss = F.cross_entropy(logits, labels, reduction='mean')
    total_loss += loss.item()
    num_batches += 1
  return total_loss / max(num_batches, 1)


def print_results_summary(row):
  """실험 결과를 터미널/주피터 셀에 보기 좋게 출력."""
  print("\n" + "=" * 52)
  print(f"  [RESULT] {row['experiment']}  "
        f"(model={row['model_size']}, last_k={row['last_k']}, lr={row['lr']})")
  print("-" * 52)
  print(f"  Dev Loss          : {row['dev_loss']}")
  print(f"  chrF              : {row['chrF']}")
  print(f"  Trainable Params  : {row['trainable_params']:,} / {row['total_params']:,} "
        f"({row['trainable_pct']}%)")
  print(f"  Train Time (sec)  : {row['train_time_sec']}")
  print(f"  Gen Time (sec)    : {row['gen_time_sec']}")
  print("=" * 52 + "\n")
  # 표에 그대로 붙여넣기 좋은 한 줄(탭 구분)도 같이 출력.
  print("TSV\t" + "\t".join(str(row[k]) for k in [
    'experiment', 'last_k', 'lr', 'dev_loss', 'chrF',
    'trainable_params', 'train_time_sec', 'gen_time_sec']))


def train(args):
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련.""" 
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if args.log_path is not None:
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "w", encoding="utf-8") as f:
        f.write("SonnetGPT Training Log\n")
        f.write("======================\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"model_size: {args.model_size}\n")
        f.write(f"temperature: {args.temperature}\n")
        f.write(f"top_p: {args.top_p}\n")
        f.write(f"sonnet_path: {args.sonnet_path}\n")
        f.write(f"held_out_sonnet_path: {args.held_out_sonnet_path}\n\n")
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  # Last-k 학습 시 requires_grad=True 인 파라미터만 옵티마이저에 전달.
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = AdamW(trainable_params, lr=lr)

  # Trainable Params 표 기록용 로깅.
  n_trainable = sum(p.numel() for p in trainable_params)
  n_total = sum(p.numel() for p in model.parameters())
  write_log(
    f"last_k: {getattr(args, 'last_k', 0)} | "
    f"Trainable params: {n_trainable:,} / {n_total:,} "
    f"({100.0 * n_trainable / n_total:.3f}%)",
    args.log_path,
  )

  train_start = time.time()
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # 시퀀스의 마지막 예측은 무시한다.
      labels = b_ids[:, 1:].contiguous().flatten()  # 레이블을 구성하기 위해 첫번째 토큰을 무시한다.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    write_log(f"Epoch {epoch}: train loss :: {train_loss :.3f}.", args.log_path)
    write_log('Generating several output sonnets...', args.log_path)
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')
      write_log(f'{batch[1]}{output[1]}\n\n', args.log_path)

    # TODO: 소넷의 작은 테이터셋에서 과적합을 방지하기 위한 종료 조건을 생각하시오.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

  train_time = time.time() - train_start
  write_log(f"Train time (sec): {train_time:.1f}", args.log_path)
  return {
    'train_time_sec': round(train_time, 1),
    'trainable_params': n_trainable,
    'total_params': n_total,
    'trainable_pct': round(100.0 * n_trainable / n_total, 4),
  }


@torch.no_grad()
def generate_submission_sonnets(args, train_stats=None):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Dev Loss: dev gold(전체 14줄) 소넷에 대한 teacher-forced cross-entropy.
  dev_loss = compute_dev_loss(model, args.gold_path, device, batch_size=args.batch_size)
  write_log(f"Dev loss: {dev_loss:.4f}", args.log_path)

  # held-out(dev) 데이터셋: 처음 3 줄만 있다. 나머지를 생성한다.
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  gen_start = time.time()
  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature,
                            top_p=args.top_p, max_length=args.max_length)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')
    write_log(f'{decoded_output}\n\n', args.log_path)
  gen_time = time.time() - gen_start

  with open(args.sonnet_out, "w+", encoding="utf-8") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

  # 생성물 vs dev gold 로 chrF 계산 (evaluation.test_sonnet 재사용, sacrebleu).
  chrf = test_sonnet(test_path=args.sonnet_out, gold_path=args.gold_path)
  write_log(f"chrF: {chrf:.4f} | gen_time(sec): {gen_time:.1f}", args.log_path)

  # Trainable params (저장된 모델에서 직접 카운트).
  n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  n_total = sum(p.numel() for p in model.parameters())
  if train_stats is None:
    train_stats = {'train_time_sec': '', 'trainable_params': n_trainable,
                   'total_params': n_total,
                   'trainable_pct': round(100.0 * n_trainable / n_total, 4)}

  row = {
    'experiment': args.exp_tag,
    'model_size': args.model_size,
    'last_k': args.last_k,
    'lr': args.lr,
    'epochs': args.epochs,
    'dev_loss': round(dev_loss, 4),
    'chrF': round(chrf, 4),
    'trainable_params': train_stats['trainable_params'],
    'total_params': train_stats['total_params'],
    'trainable_pct': train_stats['trainable_pct'],
    'train_time_sec': train_stats['train_time_sec'],
    'gen_time_sec': round(gen_time, 1),
  }
  print_results_summary(row)
  return row


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")  # train
  parser.add_argument("--held_out_sonnet_path", type=str,
                      default="data/sonnets_held_out_dev.txt")  # dev 프롬프트(첫 3줄)
  parser.add_argument("--gold_path", type=str,
                      default="data/TRUE_sonnets_held_out_dev.txt")  # dev 정답(전체 14줄). test 소넷 사용 금지.
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--exp_tag", type=str, default="exp")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--max_length", type=int, help="max generation length.", default=180)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--last_k", type=int, default=0,
                      help="Fine-tune only the last k transformer blocks (last-k). 0 = full fine-tuning.")
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  
  parser.add_argument("--log_path", type=str, default="logs/sonnet_train.log")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  elif args.model_size == 'gpt2-xl':
    args.d = 1600
    args.l = 48
    args.num_heads = 25
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  # K1/K2/K3 처럼 lr이 같고 last_k만 다른 실험이 서로 덮어쓰지 않도록 파일명에 last_k 포함.
  args.filepath = f'{args.model_size}-lastk{args.last_k}-{args.epochs}-{args.lr}-sonnet.pt'
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train_stats = train(args)
  generate_submission_sonnets(args, train_stats)