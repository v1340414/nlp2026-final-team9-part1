import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # key, value, query에 대한 선형변환 layer 초기화.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # 이 드롭아웃은 트랜스포머의 원래 구현에 따라 normalized attention scores에 적용된다.
    # 다소 이례적이지만, 경험적으로 이것이 더 나은 성능을 제공한다고 알려져 있다.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
    proj = linear_layer(x)
    # 다음으로, 프로젝션에 대해 여러 헤드를 생성해야 한다. 
    # 이는 은닉 상태를 self.num_attention_heads로 분할하며, 
    # 각 헤드는 self.attention_head_size 크기를 갖도록 한다.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # 적절히 전치하여 크기 [bs, num_attention_heads, seq_len, attention_head_size]인 프로젝션을 얻는다.
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### 완성시켜야 할 빈 코드 블록
    # 입력 Tensor shape
    # key, query, value: [bs, num_heads, seq_len, head_dim]
    # attention_mask: [bs, 1,1, seq_len]
    bs, num_heads, seq_len, head_dim = query.size()
    
    # 1. score 계산 -> Q랑 K 내적
    score = query @ key.transpose(-1,-2)
    
    # 2. sqrt(head_dim)으로 scaling
    score = score / (head_dim ** 0.5)
    
    # 3. casual mask 생성(하삼각행렬)
    casual_mask = torch.triu(
      torch.ones(seq_len, seq_len, device=score.device, dtype=torch.bool), diagonal = 1,
    )
    
    # 가리는 곳(true 위치)에 -10000 더함
    score = score.masked_fill(casual_mask, -10000.0)
    
    # 4. padding mask 적용
    score = score + attention_mask
    
    # 5. softmax
    attention_prob = torch.softmax(score, dim = -1)
    
    # 6. Dropout
    attention_prob = self.dropout(attention_prob)
    
    # 7. value에 attention 가중치 적용
    context = attention_prob @ value
    
    # 8. multi-head 다시 합쳐서 [bs, seq_len, hidden_size] 모양으로 만들기
    context = context.transpose(1,2).contiguous()
    context = context.view(bs, seq_len, num_heads * head_dim)
    
    return context
    

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # 먼저, self.transform을 사용하여 multi-head attention에 필요한
    # 각 토큰의 key, value, query를 생성해야 한다(함수 내부에 자세한 내용 있음).
    # *_layer의 크기 = [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # multi-head attention 계산.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
