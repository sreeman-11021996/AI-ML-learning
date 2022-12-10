from sequential import sequential_
import pandas as pd
import numpy as np

# Data Frame
df = pd.DataFrame([[8,8,4],[7,9,5],[6,10,6],[5,12,7]],
                  columns=['cgpa', 'profile_score', 'lpa'])
X = df[['cgpa', 'profile_score']].values[0].reshape(2,1)
y = df[['lpa']].values[0][0]

model = sequential_()

# Hidden layer 1
model.add(nodes=2,input_dim=2)
# Output Layer
model.add(nodes=1)

# Compile
model.compile()
# Forward Prop
y_pred = model.forward_prop_(X)
print("X - ",X)
print("y - ",y)
#print(model.params_)
print("y_pred - ", y_pred)
print("Outputs - ",model.outputs)

print("w1(11) - ", model.params_["w1"][1-1][1-1])


