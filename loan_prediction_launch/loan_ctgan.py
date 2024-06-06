import pandas as pd
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import RegularSynthesizer

# Load data and define the data processor parameters
file_path = "loan_input.csv"
data = pd.read_csv(file_path)

cat_cols = ['Grade', 'Sub Grade', 'HomeOwnership', 'Verification Status', 'Initial List Status']
num_cols = ['ID', 'Amount', 'Funded Amount', 'Funded Amount Investor', 'Term', 'Interest Rate',
            'Salary', 'Debit to Income', 'Inquires - six months', 'Open Account', 'Revolving Balance',
            'Total Accounts', 'Total Received Interest', 'Total Received Late Fee', 'Recoveries',
            'Collection Recovery Fee', 'Last week Pay', 'Total Collection Amount', 'Balance', 'Defaulted']

# Defining the training parameters
batch_size = 500
epochs = 500+1
learning_rate = 2e-4
beta_1 = 0.5
beta_2 = 0.9

ctgan_args = ModelParameters(batch_size=batch_size,
                             lr=learning_rate,
                             betas=(beta_1, beta_2))

train_args = TrainParameters(epochs=epochs)
synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)


# Saving the synthesizer
synth.save('loan_model.pkl')

# Loading the synthesizer
synth = RegularSynthesizer.load('loan_model.pkl')

# Loading and sampling from a trained synthesizer
synth_data = synth.sample(24000)
print(synth_data)
synth_data.to_csv('loan_result_3.csv', index=False)
