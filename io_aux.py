
def read_csv(file):
    import pandas as pd
    inputs = pd.read_csv(file, delimiter=";", comment='#')
    if (inputs.weight != inputs.amount/inputs.amount.sum()).any():
        inputs.weight = (inputs.amount/inputs.amount.sum()).round(4) # weights
        inputs.to_csv('inputs.csv', index=False, sep=';', 
                      mode='w')
    return inputs







