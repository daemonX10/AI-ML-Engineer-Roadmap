# Tinygrad Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Create a Float Tensor with Tinygrad.**

### Answer
```python
from tinygrad import Tensor, dtypes

def to_float_tensor(values):
    """Convert list or list-of-lists to a float32 Tinygrad Tensor."""
    return Tensor(values, dtype=dtypes.float32)

# Example
t1 = to_float_tensor([1, 2, 3])
t2 = to_float_tensor([[1, 2], [3, 4]])
print(t1, t1.dtype)
print(t2, t2.dtype)
```
