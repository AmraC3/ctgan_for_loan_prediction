import pandas as pd
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Load data and define the data processor parameters
file_path = "loan_input.csv"
data = pd.read_csv(file_path)
num_cols = ['ID', 'Amount', 'Funded Amount', 'Funded Amount Investor', 'Term', 'Interest Rate',
            'Salary', 'Debit to Income', 'Inquires - six months', 'Open Account', 'Revolving Balance',
            'Total Accounts', 'Total Received Interest', 'Total Received Late Fee', 'Recoveries',
            'Collection Recovery Fee', 'Last week Pay', 'Total Collection Amount', 'Balance', 'Defaulted']
cat_cols = ['Grade', 'Sub Grade', 'HomeOwnership', 'Verification Status', 'Initial List Status']

# DRAGAN training
#Defining the training parameters of DRAGAN
noise_dim = 128
dim = 128
batch_size = 500

log_step = 100
epochs = 500+1
learning_rate = 1e-5
beta_1 = 0.5
beta_2 = 0.9
models_dir = '../cache'

gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             sample_interval=log_step)

synth = RegularSynthesizer(modelname='dragan', model_parameters=gan_args, n_discriminator=3)
synth.fit(data = data, train_arguments = train_args, num_cols = num_cols, cat_cols = cat_cols)

synth.save('loan_dragan_model.pkl')

#########################################################
#    Loading and sampling from a trained synthesizer    #
#########################################################
synthesizer = RegularSynthesizer.load('loan_dragan_model.pkl')
synth_data = synthesizer.sample(24000)
print(synth_data)
synth_data.to_csv('dragan_loan_result.csv', index=False)
