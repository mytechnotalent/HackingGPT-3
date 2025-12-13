![image](https://github.com/mytechnotalent/HackingGPT-3/blob/main/HackingGPT.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# HackingGPT
## Part 3
Part 3 covers naive averaging with for loops, "bag of words" token aggregation, and understanding why Python loops are slow before optimizing with matrix multiplication.

#### Author: [Kevin Thomas](mailto:ket189@pitt.edu)

<br>

## Part 2 [HERE](https://github.com/mytechnotalent/HackingGPT-2)

<br><br>

```python
import torch
```


## Step 1: Load and Inspect the Data
Now let's read the file and see what we're working with. Understanding your data is crucial before building any model!


```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```


```python
text
```


**Output:**
```
'A dim glow rises behind the glass of a screen and the machine exhales in binary tides. The hum is a language and one who listens leans close to catch the quiet grammar. Patterns fold like small maps and seams hint at how the thing holds itself together. Treat each blinking diode and each idle tick as a sentence in a story that asks to be read.\n\nThere is patience here, not of haste but of careful unthreading. Where others see a sealed box the curious hand traces the join and wonders which thought made it fit. Do not rush to break, coax the meaning out with questions, and watch how the logic replies in traces and errors and in the echoes of forgotten interfaces.\n\nTechnology is artifact and argument at once. It makes a claim about what should be simple, what should be hidden, and what should be trusted. Reverse the gaze and learn its rhetoric, see where it promises ease, where it buries complexity, and where it leaves a backdoor as a sigh between bricks. To read that rhetoric is to be a kind interpreter, not a vandal.\n\nThis work is an apprenticeship in humility. Expect bafflement and expect to be corrected by small things, a timing oddity, a mismatch of expectation, a choice that favors speed over grace. Each misstep teaches a vocabulary of trade offs. Each discovery is a map of decisions and not a verdict on worth.\n\nThere is a moral keeping in the craft. Let curiosity be tempered with regard for consequence. Let repair and understanding lead rather than exploitation. The skill that opens a lock should also know when to hold the key and when to hand it back, mindful of harm and mindful of help.\n\nCelebrate the quiet victories, a stubborn protocol understood, an obscure format rendered speakable, a closed device coaxed into cooperation. These are small reconciliations between human intent and metal will, acts of translation rather than acts of conquest.\n\nAfter decoding a mechanism pause and ask what should change, a bug to be fixed, a user to be warned, a design to be amended. The true maker of machines leaves things better for having looked, not simply for having cracked the shell.'
```


## Step 2: Version 1 - Naive Averaging (For Loops)
**Goal**: For each position `t`, compute the mean of all positions up to and including `t`.

This is "bag of words" style where we average the past tokens, losing their order.

### What does "averaging previous positions" mean?
Imagine you have a sequence of 8 tokens. At each position, we want to gather information from all previous tokens.
| Position | What it sees | Number of tokens averaged |
|----------|--------------|---------------------------|
| 0 | just itself | 1 |
| 1 | positions 0, 1 | 2 |
| 2 | positions 0, 1, 2 | 3 |
| 3 | positions 0, 1, 2, 3 | 4 |
| 4 | positions 0, 1, 2, 3, 4 | 5 |
| 5 | positions 0, 1, 2, 3, 4, 5 | 6 |
| 6 | positions 0, 1, 2, 3, 4, 5, 6 | 7 |
| 7 | positions 0, 1, 2, 3, 4, 5, 6, 7 | 8 |

### Why is this called "bag of words"?
When we average, we lose the order of the tokens. Position 0 coming first and position 2 coming last gives the same average as position 2 coming first and position 0 coming last. The tokens are thrown into a "bag" and mixed together.

### Why would we want this?
This is the simplest form of "communication" between tokens. Each token gets to see what came before it. Later, we'll make this smarter with attention, where tokens can decide how much to look at each previous token instead of giving them all equal weight.


```python
torch.manual_seed(42)
```


**Output:**
```
<torch._C.Generator at 0x125c19a10>
```


```python
# define batch dimension
B = 4  # batch size: 4 independent sequences
B
```


**Output:**
```
4
```


```python
# define time dimension
T = 8  # sequence length: 8 tokens/positions in each sequence
T
```


**Output:**
```
8
```


```python
# define channel dimension
C = 2  # feature size: 2 features per token
C
```


**Output:**
```
2
```


```python
# start with random data
x = torch.randn(B, T, C)
x
```


**Output:**
```
tensor([[[ 1.9269,  1.4873],
         [ 0.9007, -2.1055],
         [ 0.6784, -1.2345],
         [-0.0431, -1.6047],
         [-0.7521,  1.6487],
         [-0.3925, -1.4036],
         [-0.7279, -0.5594],
         [-0.7688,  0.7624]],

        [[ 1.6423, -0.1596],
         [-0.4974,  0.4396],
         [-0.7581,  1.0783],
         [ 0.8008,  1.6806],
         [ 1.2791,  1.2964],
         [ 0.6105,  1.3347],
         [-0.2316,  0.0418],
         [-0.2516,  0.8599]],

        [[-1.3847, -0.8712],
         [-0.2234,  1.7174],
         [ 0.3189, -0.4245],
         [ 0.3057, -0.7746],
         [-1.5576,  0.9956],
         [-0.8798, -0.6011],
         [-1.2742,  2.1228],
         [-1.2347, -0.4879]],

        [[-0.9138, -0.6581],
         [ 0.0780,  0.5258],
         [-0.4880,  1.1914],
         [-0.8140, -0.7360],
         [-1.4032,  0.0360],
         [-0.0635,  0.6756],
         [-0.0978,  1.8446],
         [-1.1845,  1.3835]]])
```


```python
# "bow" = bag of words (averaging)
x_bow = torch.zeros((B, T, C))  
x_bow
```


**Output:**
```
tensor([[[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]]])
```


```python
# each position averages all previous positions (including itself)
# loop over batches
for b in range(B):    
    # loop over positions       
    for t in range(T):      
        # all positions from 0 to t (inclusive) 
        # shape: (t+1, C)
        x_previous = x[b, :t+1]
        print(f'batch {b}, position {t}: x_previous: {x_previous}')
        print(f'batch {b}, position {t}: x_previous shape: {x_previous.shape}') 
        # average them → shape (C,)
        x_bow[b, t] = torch.mean(x_previous, dim=0)  
        print(f'batch {b}, position {t}: x_bow[b, t]: {x_bow[b, t]}')
        print()
```


**Output:**
```
batch 0, position 0: x_previous: tensor([[1.9269, 1.4873]])
batch 0, position 0: x_previous shape: torch.Size([1, 2])
batch 0, position 0: x_bow[b, t]: tensor([1.9269, 1.4873])

batch 0, position 1: x_previous: tensor([[ 1.9269,  1.4873],
        [ 0.9007, -2.1055]])
batch 0, position 1: x_previous shape: torch.Size([2, 2])
batch 0, position 1: x_bow[b, t]: tensor([ 1.4138, -0.3091])

batch 0, position 2: x_previous: tensor([[ 1.9269,  1.4873],
        [ 0.9007, -2.1055],
        [ 0.6784, -1.2345]])
batch 0, position 2: x_previous shape: torch.Size([3, 2])
batch 0, position 2: x_bow[b, t]: tensor([ 1.1687, -0.6176])

batch 0, position 3: x_previous: tensor([[ 1.9269,  1.4873],
        [ 0.9007, -2.1055],
        [ 0.6784, -1.2345],
        [-0.0431, -1.6047]])
batch 0, position 3: x_previous shape: torch.Size([4, 2])
batch 0, position 3: x_bow[b, t]: tensor([ 0.8657, -0.8644])

batch 0, position 4: x_previous: tensor([[ 1.9269,  1.4873],
        [ 0.9007, -2.1055],
        [ 0.6784, -1.2345],
        [-0.0431, -1.6047],
        [-0.7521,  1.6487]])
batch 0, position 4: x_previous shape: torch.Size([5, 2])
batch 0, position 4: x_bow[b, t]: tensor([ 0.5422, -0.3617])

batch 0, position 5: x_previous: tensor([[ 1.9269,  1.4873],
        [ 0.9007, -2.1055],
        [ 0.6784, -1.2345],
        [-0.0431, -1.6047],
        [-0.7521,  1.6487],
        [-0.3925, -1.4036]])
batch 0, position 5: x_previous shape: torch.Size([6, 2])
batch 0, position 5: x_bow[b, t]: tensor([ 0.3864, -0.5354])

batch 0, position 6: x_previous: tensor([[ 1.9269,  1.4873],
        [ 0.9007, -2.1055],
        [ 0.6784, -1.2345],
        [-0.0431, -1.6047],
        [-0.7521,  1.6487],
        [-0.3925, -1.4036],
        [-0.7279, -0.5594]])
batch 0, position 6: x_previous shape: torch.Size([7, 2])
batch 0, position 6: x_bow[b, t]: tensor([ 0.2272, -0.5388])

batch 0, position 7: x_previous: tensor([[ 1.9269,  1.4873],
        [ 0.9007, -2.1055],
        [ 0.6784, -1.2345],
        [-0.0431, -1.6047],
        [-0.7521,  1.6487],
        [-0.3925, -1.4036],
        [-0.7279, -0.5594],
        [-0.7688,  0.7624]])
batch 0, position 7: x_previous shape: torch.Size([8, 2])
batch 0, position 7: x_bow[b, t]: tensor([ 0.1027, -0.3762])

batch 1, position 0: x_previous: tensor([[ 1.6423, -0.1596]])
batch 1, position 0: x_previous shape: torch.Size([1, 2])
batch 1, position 0: x_bow[b, t]: tensor([ 1.6423, -0.1596])

batch 1, position 1: x_previous: tensor([[ 1.6423, -0.1596],
        [-0.4974,  0.4396]])
batch 1, position 1: x_previous shape: torch.Size([2, 2])
batch 1, position 1: x_bow[b, t]: tensor([0.5725, 0.1400])

batch 1, position 2: x_previous: tensor([[ 1.6423, -0.1596],
        [-0.4974,  0.4396],
        [-0.7581,  1.0783]])
batch 1, position 2: x_previous shape: torch.Size([3, 2])
batch 1, position 2: x_bow[b, t]: tensor([0.1289, 0.4528])

batch 1, position 3: x_previous: tensor([[ 1.6423, -0.1596],
        [-0.4974,  0.4396],
        [-0.7581,  1.0783],
        [ 0.8008,  1.6806]])
batch 1, position 3: x_previous shape: torch.Size([4, 2])
batch 1, position 3: x_bow[b, t]: tensor([0.2969, 0.7597])

batch 1, position 4: x_previous: tensor([[ 1.6423, -0.1596],
        [-0.4974,  0.4396],
        [-0.7581,  1.0783],
        [ 0.8008,  1.6806],
        [ 1.2791,  1.2964]])
batch 1, position 4: x_previous shape: torch.Size([5, 2])
batch 1, position 4: x_bow[b, t]: tensor([0.4933, 0.8671])

batch 1, position 5: x_previous: tensor([[ 1.6423, -0.1596],
        [-0.4974,  0.4396],
        [-0.7581,  1.0783],
        [ 0.8008,  1.6806],
        [ 1.2791,  1.2964],
        [ 0.6105,  1.3347]])
batch 1, position 5: x_previous shape: torch.Size([6, 2])
batch 1, position 5: x_bow[b, t]: tensor([0.5129, 0.9450])

batch 1, position 6: x_previous: tensor([[ 1.6423, -0.1596],
        [-0.4974,  0.4396],
        [-0.7581,  1.0783],
        [ 0.8008,  1.6806],
        [ 1.2791,  1.2964],
        [ 0.6105,  1.3347],
        [-0.2316,  0.0418]])
batch 1, position 6: x_previous shape: torch.Size([7, 2])
batch 1, position 6: x_bow[b, t]: tensor([0.4065, 0.8160])

batch 1, position 7: x_previous: tensor([[ 1.6423, -0.1596],
        [-0.4974,  0.4396],
        [-0.7581,  1.0783],
        [ 0.8008,  1.6806],
        [ 1.2791,  1.2964],
        [ 0.6105,  1.3347],
        [-0.2316,  0.0418],
        [-0.2516,  0.8599]])
batch 1, position 7: x_previous shape: torch.Size([8, 2])
batch 1, position 7: x_bow[b, t]: tensor([0.3242, 0.8215])

batch 2, position 0: x_previous: tensor([[-1.3847, -0.8712]])
batch 2, position 0: x_previous shape: torch.Size([1, 2])
batch 2, position 0: x_bow[b, t]: tensor([-1.3847, -0.8712])

batch 2, position 1: x_previous: tensor([[-1.3847, -0.8712],
        [-0.2234,  1.7174]])
batch 2, position 1: x_previous shape: torch.Size([2, 2])
batch 2, position 1: x_bow[b, t]: tensor([-0.8040,  0.4231])

batch 2, position 2: x_previous: tensor([[-1.3847, -0.8712],
        [-0.2234,  1.7174],
        [ 0.3189, -0.4245]])
batch 2, position 2: x_previous shape: torch.Size([3, 2])
batch 2, position 2: x_bow[b, t]: tensor([-0.4297,  0.1405])

batch 2, position 3: x_previous: tensor([[-1.3847, -0.8712],
        [-0.2234,  1.7174],
        [ 0.3189, -0.4245],
        [ 0.3057, -0.7746]])
batch 2, position 3: x_previous shape: torch.Size([4, 2])
batch 2, position 3: x_bow[b, t]: tensor([-0.2459, -0.0882])

batch 2, position 4: x_previous: tensor([[-1.3847, -0.8712],
        [-0.2234,  1.7174],
        [ 0.3189, -0.4245],
        [ 0.3057, -0.7746],
        [-1.5576,  0.9956]])
batch 2, position 4: x_previous shape: torch.Size([5, 2])
batch 2, position 4: x_bow[b, t]: tensor([-0.5082,  0.1285])

batch 2, position 5: x_previous: tensor([[-1.3847, -0.8712],
        [-0.2234,  1.7174],
        [ 0.3189, -0.4245],
        [ 0.3057, -0.7746],
        [-1.5576,  0.9956],
        [-0.8798, -0.6011]])
batch 2, position 5: x_previous shape: torch.Size([6, 2])
batch 2, position 5: x_bow[b, t]: tensor([-0.5701,  0.0069])

batch 2, position 6: x_previous: tensor([[-1.3847, -0.8712],
        [-0.2234,  1.7174],
        [ 0.3189, -0.4245],
        [ 0.3057, -0.7746],
        [-1.5576,  0.9956],
        [-0.8798, -0.6011],
        [-1.2742,  2.1228]])
batch 2, position 6: x_previous shape: torch.Size([7, 2])
batch 2, position 6: x_bow[b, t]: tensor([-0.6707,  0.3092])

batch 2, position 7: x_previous: tensor([[-1.3847, -0.8712],
        [-0.2234,  1.7174],
        [ 0.3189, -0.4245],
        [ 0.3057, -0.7746],
        [-1.5576,  0.9956],
        [-0.8798, -0.6011],
        [-1.2742,  2.1228],
        [-1.2347, -0.4879]])
batch 2, position 7: x_previous shape: torch.Size([8, 2])
batch 2, position 7: x_bow[b, t]: tensor([-0.7412,  0.2095])

batch 3, position 0: x_previous: tensor([[-0.9138, -0.6581]])
batch 3, position 0: x_previous shape: torch.Size([1, 2])
batch 3, position 0: x_bow[b, t]: tensor([-0.9138, -0.6581])

batch 3, position 1: x_previous: tensor([[-0.9138, -0.6581],
        [ 0.0780,  0.5258]])
batch 3, position 1: x_previous shape: torch.Size([2, 2])
batch 3, position 1: x_bow[b, t]: tensor([-0.4179, -0.0662])

batch 3, position 2: x_previous: tensor([[-0.9138, -0.6581],
        [ 0.0780,  0.5258],
        [-0.4880,  1.1914]])
batch 3, position 2: x_previous shape: torch.Size([3, 2])
batch 3, position 2: x_bow[b, t]: tensor([-0.4413,  0.3530])

batch 3, position 3: x_previous: tensor([[-0.9138, -0.6581],
        [ 0.0780,  0.5258],
        [-0.4880,  1.1914],
        [-0.8140, -0.7360]])
batch 3, position 3: x_previous shape: torch.Size([4, 2])
batch 3, position 3: x_bow[b, t]: tensor([-0.5344,  0.0808])

batch 3, position 4: x_previous: tensor([[-0.9138, -0.6581],
        [ 0.0780,  0.5258],
        [-0.4880,  1.1914],
        [-0.8140, -0.7360],
        [-1.4032,  0.0360]])
batch 3, position 4: x_previous shape: torch.Size([5, 2])
batch 3, position 4: x_bow[b, t]: tensor([-0.7082,  0.0718])

batch 3, position 5: x_previous: tensor([[-0.9138, -0.6581],
        [ 0.0780,  0.5258],
        [-0.4880,  1.1914],
        [-0.8140, -0.7360],
        [-1.4032,  0.0360],
        [-0.0635,  0.6756]])
batch 3, position 5: x_previous shape: torch.Size([6, 2])
batch 3, position 5: x_bow[b, t]: tensor([-0.6008,  0.1724])

batch 3, position 6: x_previous: tensor([[-0.9138, -0.6581],
        [ 0.0780,  0.5258],
        [-0.4880,  1.1914],
        [-0.8140, -0.7360],
        [-1.4032,  0.0360],
        [-0.0635,  0.6756],
        [-0.0978,  1.8446]])
batch 3, position 6: x_previous shape: torch.Size([7, 2])
batch 3, position 6: x_bow[b, t]: tensor([-0.5289,  0.4113])

batch 3, position 7: x_previous: tensor([[-0.9138, -0.6581],
        [ 0.0780,  0.5258],
        [-0.4880,  1.1914],
        [-0.8140, -0.7360],
        [-1.4032,  0.0360],
        [-0.0635,  0.6756],
        [-0.0978,  1.8446],
        [-1.1845,  1.3835]])
batch 3, position 7: x_previous shape: torch.Size([8, 2])
batch 3, position 7: x_bow[b, t]: tensor([-0.6109,  0.5329])


```


```python
print('version 1: naive for-loop averaging')
print()
print(f'input shape:  {x.shape} → (B={B}, T={T}, C={C})')
print(f'output shape: {x_bow.shape} → (B={B}, T={T}, C={C})')
print()
print('Same shape! Each position now holds the average of itself and all previous positions.')
```


**Output:**
```
version 1: naive for-loop averaging

input shape:  torch.Size([4, 8, 2]) → (B=4, T=8, C=2)
output shape: torch.Size([4, 8, 2]) → (B=4, T=8, C=2)

Same shape! Each position now holds the average of itself and all previous positions.

```


```python
print('example: batch 0, position 0')
print('position 0 averages tokens 0 (all positions up to and including itself)')
print()
print('token values:')
print(f'   x[0, 0] = {x[0, 0].tolist()}')
print()
print('calculation:')
print(f'   mean = ({x[0, 0].tolist()}) / 1')
print(f'        =  {x_bow[0, 0].tolist()}')
print()
print('verify:')
print(f'   x_bow[0, 0]          = {x_bow[0, 0].tolist()}')
print(f'   torch.mean(x[0, :1]) = {torch.mean(x[0, :1], dim=0).tolist()}')
```


**Output:**
```
example: batch 0, position 0
position 0 averages tokens 0 (all positions up to and including itself)

token values:
   x[0, 0] = [1.9269150495529175, 1.4872841835021973]

calculation:
   mean = ([1.9269150495529175, 1.4872841835021973]) / 1
        =  [1.9269150495529175, 1.4872841835021973]

verify:
   x_bow[0, 0]          = [1.9269150495529175, 1.4872841835021973]
   torch.mean(x[0, :1]) = [1.9269150495529175, 1.4872841835021973]

```


```python
print('example: batch 0, position 1')
print('position 1 averages tokens 0 and 1 (all positions up to and including itself)')
print()
print('token values:')
print(f'   x[0, 0] = {x[0, 0].tolist()}')
print(f'   x[0, 1] = {x[0, 1].tolist()}')
print()
print('calculation:')
print(f'   mean = ({x[0, 0].tolist()}')
print(f'        +  {x[0, 1].tolist()}) / 2')
print(f'        =  {x_bow[0, 1].tolist()}')
print()
print('verify:')
print(f'   x_bow[0, 1]         = {x_bow[0, 1].tolist()}')
print(f'   torch.mean(x[0,:2]) = {torch.mean(x[0, :2], dim=0).tolist()}')
```


**Output:**
```
example: batch 0, position 1
position 1 averages tokens 0 and 1 (all positions up to and including itself)

token values:
   x[0, 0] = [1.9269150495529175, 1.4872841835021973]
   x[0, 1] = [0.9007171988487244, -2.1055214405059814]

calculation:
   mean = ([1.9269150495529175, 1.4872841835021973]
        +  [0.9007171988487244, -2.1055214405059814]) / 2
        =  [1.4138160943984985, -0.3091186285018921]

verify:
   x_bow[0, 1]         = [1.4138160943984985, -0.3091186285018921]
   torch.mean(x[0,:2]) = [1.4138160943984985, -0.3091186285018921]

```


```python
print('example: batch 0, position 2')
print('position 2 averages tokens 0, 1, and 2 (all positions up to and including itself)')
print()
print('token values:')
print(f'   x[0, 0] = {x[0, 0].tolist()}')
print(f'   x[0, 1] = {x[0, 1].tolist()}')
print(f'   x[0, 2] = {x[0, 2].tolist()}')
print()
print('calculation:')
print(f'   mean = ({x[0, 0].tolist()}')
print(f'        +  {x[0, 1].tolist()}')
print(f'        +  {x[0, 2].tolist()}) / 3')
print(f'        =  {x_bow[0, 2].tolist()}')
print()
print('verify:')
print(f'   x_bow[0, 2]         = {x_bow[0, 2].tolist()}')
print(f'   torch.mean(x[0,:3]) = {torch.mean(x[0, :3], dim=0).tolist()}')
```


**Output:**
```
example: batch 0, position 2
position 2 averages tokens 0, 1, and 2 (all positions up to and including itself)

token values:
   x[0, 0] = [1.9269150495529175, 1.4872841835021973]
   x[0, 1] = [0.9007171988487244, -2.1055214405059814]
   x[0, 2] = [0.6784184575080872, -1.2345449924468994]

calculation:
   mean = ([1.9269150495529175, 1.4872841835021973]
        +  [0.9007171988487244, -2.1055214405059814]
        +  [0.6784184575080872, -1.2345449924468994]) / 3
        =  [1.1686835289001465, -0.6175940632820129]

verify:
   x_bow[0, 2]         = [1.1686835289001465, -0.6175940632820129]
   torch.mean(x[0,:3]) = [1.1686835289001465, -0.6175940632820129]

```


```python
print('example: batch 0, position 3')
print('position 3 averages tokens 0, 1, 2, and 3 (all positions up to and including itself)')
print()
print('token values:')
print(f'   x[0, 0] = {x[0, 0].tolist()}')
print(f'   x[0, 1] = {x[0, 1].tolist()}')
print(f'   x[0, 2] = {x[0, 2].tolist()}')
print(f'   x[0, 3] = {x[0, 3].tolist()}')
print()
print('calculation:')
print(f'   mean = ({x[0, 0].tolist()}')
print(f'        +  {x[0, 1].tolist()}')
print(f'        +  {x[0, 2].tolist()}')
print(f'        +  {x[0, 3].tolist()}) / 4')
print(f'        =  {x_bow[0, 3].tolist()}')
print()
print('verify:')
print(f'   x_bow[0, 3]         = {x_bow[0, 3].tolist()}')
print(f'   torch.mean(x[0,:4]) = {torch.mean(x[0, :4], dim=0).tolist()}')
```


**Output:**
```
example: batch 0, position 3
position 3 averages tokens 0, 1, 2, and 3 (all positions up to and including itself)

token values:
   x[0, 0] = [1.9269150495529175, 1.4872841835021973]
   x[0, 1] = [0.9007171988487244, -2.1055214405059814]
   x[0, 2] = [0.6784184575080872, -1.2345449924468994]
   x[0, 3] = [-0.043067481368780136, -1.6046669483184814]

calculation:
   mean = ([1.9269150495529175, 1.4872841835021973]
        +  [0.9007171988487244, -2.1055214405059814]
        +  [0.6784184575080872, -1.2345449924468994]
        +  [-0.043067481368780136, -1.6046669483184814]) / 4
        =  [0.8657457828521729, -0.8643622994422913]

verify:
   x_bow[0, 3]         = [0.8657457828521729, -0.8643622994422913]
   torch.mean(x[0,:4]) = [0.8657457828521729, -0.8643622994422913]

```


```python
print('example: batch 0, position 4')
print('position 4 averages tokens 0, 1, 2, 3, and 4 (all positions up to and including itself)')
print()
print('token values:')
print(f'   x[0, 0] = {x[0, 0].tolist()}')
print(f'   x[0, 1] = {x[0, 1].tolist()}')
print(f'   x[0, 2] = {x[0, 2].tolist()}')
print(f'   x[0, 3] = {x[0, 3].tolist()}')
print(f'   x[0, 4] = {x[0, 4].tolist()}')
print()
print('calculation:')
print(f'   mean = ({x[0, 0].tolist()}')
print(f'        +  {x[0, 1].tolist()}')
print(f'        +  {x[0, 2].tolist()}')
print(f'        +  {x[0, 3].tolist()}')
print(f'        +  {x[0, 4].tolist()}) / 5')
print(f'        =  {x_bow[0, 4].tolist()}')
print()
print('verify:')
print(f'   x_bow[0, 4]         = {x_bow[0, 4].tolist()}')
print(f'   torch.mean(x[0,:5]) = {torch.mean(x[0, :5], dim=0).tolist()}')
```


**Output:**
```
example: batch 0, position 4
position 4 averages tokens 0, 1, 2, 3, and 4 (all positions up to and including itself)

token values:
   x[0, 0] = [1.9269150495529175, 1.4872841835021973]
   x[0, 1] = [0.9007171988487244, -2.1055214405059814]
   x[0, 2] = [0.6784184575080872, -1.2345449924468994]
   x[0, 3] = [-0.043067481368780136, -1.6046669483184814]
   x[0, 4] = [-0.7521361708641052, 1.6487228870391846]

calculation:
   mean = ([1.9269150495529175, 1.4872841835021973]
        +  [0.9007171988487244, -2.1055214405059814]
        +  [0.6784184575080872, -1.2345449924468994]
        +  [-0.043067481368780136, -1.6046669483184814]
        +  [-0.7521361708641052, 1.6487228870391846]) / 5
        =  [0.542169451713562, -0.36174526810646057]

verify:
   x_bow[0, 4]         = [0.542169451713562, -0.36174526810646057]
   torch.mean(x[0,:5]) = [0.542169451713562, -0.36174526810646057]

```


```python
print('example: batch 0, position 5')
print('position 5 averages tokens 0, 1, 2, 3, 4, and 5 (all positions up to and including itself)')
print()
print('token values:')
print(f'   x[0, 0] = {x[0, 0].tolist()}')
print(f'   x[0, 1] = {x[0, 1].tolist()}')
print(f'   x[0, 2] = {x[0, 2].tolist()}')
print(f'   x[0, 3] = {x[0, 3].tolist()}')
print(f'   x[0, 4] = {x[0, 4].tolist()}')
print(f'   x[0, 5] = {x[0, 5].tolist()}')
print()
print('calculation:')
print(f'   mean = ({x[0, 0].tolist()}')
print(f'        +  {x[0, 1].tolist()}')
print(f'        +  {x[0, 2].tolist()}')
print(f'        +  {x[0, 3].tolist()}')
print(f'        +  {x[0, 4].tolist()}')
print(f'        +  {x[0, 5].tolist()}) / 6')
print(f'        =  {x_bow[0, 5].tolist()}')
print()
print('verify:')
print(f'   x_bow[0, 5]         = {x_bow[0, 5].tolist()}')
print(f'   torch.mean(x[0,:6]) = {torch.mean(x[0, :6], dim=0).tolist()}')
```


**Output:**
```
example: batch 0, position 5
position 5 averages tokens 0, 1, 2, 3, 4, and 5 (all positions up to and including itself)

token values:
   x[0, 0] = [1.9269150495529175, 1.4872841835021973]
   x[0, 1] = [0.9007171988487244, -2.1055214405059814]
   x[0, 2] = [0.6784184575080872, -1.2345449924468994]
   x[0, 3] = [-0.043067481368780136, -1.6046669483184814]
   x[0, 4] = [-0.7521361708641052, 1.6487228870391846]
   x[0, 5] = [-0.3924786448478699, -1.4036067724227905]

calculation:
   mean = ([1.9269150495529175, 1.4872841835021973]
        +  [0.9007171988487244, -2.1055214405059814]
        +  [0.6784184575080872, -1.2345449924468994]
        +  [-0.043067481368780136, -1.6046669483184814]
        +  [-0.7521361708641052, 1.6487228870391846]
        +  [-0.3924786448478699, -1.4036067724227905]) / 6
        =  [0.386394739151001, -0.5353888869285583]

verify:
   x_bow[0, 5]         = [0.386394739151001, -0.5353888869285583]
   torch.mean(x[0,:6]) = [0.386394739151001, -0.5353888869285583]

```


```python
print('example: batch 0, position 6')
print('position 6 averages tokens 0, 1, 2, 3, 4, 5, and 6 (all positions up to and including itself)')
print()
print('token values:')
print(f'   x[0, 0] = {x[0, 0].tolist()}')
print(f'   x[0, 1] = {x[0, 1].tolist()}')
print(f'   x[0, 2] = {x[0, 2].tolist()}')
print(f'   x[0, 3] = {x[0, 3].tolist()}')
print(f'   x[0, 4] = {x[0, 4].tolist()}')
print(f'   x[0, 5] = {x[0, 5].tolist()}')
print(f'   x[0, 6] = {x[0, 6].tolist()}')
print()
print('calculation:')
print(f'   mean = ({x[0, 0].tolist()}')
print(f'        +  {x[0, 1].tolist()}')
print(f'        +  {x[0, 2].tolist()}')
print(f'        +  {x[0, 3].tolist()}')
print(f'        +  {x[0, 4].tolist()}')
print(f'        +  {x[0, 5].tolist()}')
print(f'        +  {x[0, 6].tolist()}) / 7')
print(f'        =  {x_bow[0, 6].tolist()}')
print()
print('verify:')
print(f'   x_bow[0, 6]         = {x_bow[0, 6].tolist()}')
print(f'   torch.mean(x[0,:7]) = {torch.mean(x[0, :7], dim=0).tolist()}')
```


**Output:**
```
example: batch 0, position 6
position 6 averages tokens 0, 1, 2, 3, 4, 5, and 6 (all positions up to and including itself)

token values:
   x[0, 0] = [1.9269150495529175, 1.4872841835021973]
   x[0, 1] = [0.9007171988487244, -2.1055214405059814]
   x[0, 2] = [0.6784184575080872, -1.2345449924468994]
   x[0, 3] = [-0.043067481368780136, -1.6046669483184814]
   x[0, 4] = [-0.7521361708641052, 1.6487228870391846]
   x[0, 5] = [-0.3924786448478699, -1.4036067724227905]
   x[0, 6] = [-0.7278812527656555, -0.5594298839569092]

calculation:
   mean = ([1.9269150495529175, 1.4872841835021973]
        +  [0.9007171988487244, -2.1055214405059814]
        +  [0.6784184575080872, -1.2345449924468994]
        +  [-0.043067481368780136, -1.6046669483184814]
        +  [-0.7521361708641052, 1.6487228870391846]
        +  [-0.3924786448478699, -1.4036067724227905]
        +  [-0.7278812527656555, -0.5594298839569092]) / 7
        =  [0.22721245884895325, -0.5388233065605164]

verify:
   x_bow[0, 6]         = [0.22721245884895325, -0.5388233065605164]
   torch.mean(x[0,:7]) = [0.22721245884895325, -0.5388233065605164]

```


```python
print('example: batch 0, position 7')
print('position 7 averages tokens 0, 1, 2, 3, 4, 5, 6, and 7 (all positions up to and including itself)')
print()
print('token values:')
print(f'   x[0, 0] = {x[0, 0].tolist()}')
print(f'   x[0, 1] = {x[0, 1].tolist()}')
print(f'   x[0, 2] = {x[0, 2].tolist()}')
print(f'   x[0, 3] = {x[0, 3].tolist()}')
print(f'   x[0, 4] = {x[0, 4].tolist()}')
print(f'   x[0, 5] = {x[0, 5].tolist()}')
print(f'   x[0, 6] = {x[0, 6].tolist()}')
print(f'   x[0, 7] = {x[0, 7].tolist()}')
print()
print('calculation:')
print(f'   mean = ({x[0, 0].tolist()}')
print(f'        +  {x[0, 1].tolist()}')
print(f'        +  {x[0, 2].tolist()}')
print(f'        +  {x[0, 3].tolist()}')
print(f'        +  {x[0, 4].tolist()}')
print(f'        +  {x[0, 5].tolist()}')
print(f'        +  {x[0, 6].tolist()}')
print(f'        +  {x[0, 7].tolist()}) / 8')
print(f'        =  {x_bow[0, 7].tolist()}')
print()
print('verify:')
print(f'   x_bow[0, 7]         = {x_bow[0, 7].tolist()}')
print(f'   torch.mean(x[0,:8]) = {torch.mean(x[0, :8], dim=0).tolist()}')
```


**Output:**
```
example: batch 0, position 7
position 7 averages tokens 0, 1, 2, 3, 4, 5, 6, and 7 (all positions up to and including itself)

token values:
   x[0, 0] = [1.9269150495529175, 1.4872841835021973]
   x[0, 1] = [0.9007171988487244, -2.1055214405059814]
   x[0, 2] = [0.6784184575080872, -1.2345449924468994]
   x[0, 3] = [-0.043067481368780136, -1.6046669483184814]
   x[0, 4] = [-0.7521361708641052, 1.6487228870391846]
   x[0, 5] = [-0.3924786448478699, -1.4036067724227905]
   x[0, 6] = [-0.7278812527656555, -0.5594298839569092]
   x[0, 7] = [-0.7688389420509338, 0.7624453902244568]

calculation:
   mean = ([1.9269150495529175, 1.4872841835021973]
        +  [0.9007171988487244, -2.1055214405059814]
        +  [0.6784184575080872, -1.2345449924468994]
        +  [-0.043067481368780136, -1.6046669483184814]
        +  [-0.7521361708641052, 1.6487228870391846]
        +  [-0.3924786448478699, -1.4036067724227905]
        +  [-0.7278812527656555, -0.5594298839569092]
        +  [-0.7688389420509338, 0.7624453902244568]) / 8
        =  [0.10270603746175766, -0.37616467475891113]

verify:
   x_bow[0, 7]         = [0.10270603746175766, -0.37616467475891113]
   torch.mean(x[0,:8]) = [0.10270603746175766, -0.37616467475891113]

```


## MIT License

