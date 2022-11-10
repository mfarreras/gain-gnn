import tensorflow as tf
import sys

sys.path.insert(1, "./code")
from read_dataset_predict import input_fn_predict
from routenet_model_predict import RouteNetModel_predict
import configparser
import os
from datanetAPI import DatanetAPI
import wandb
from wandb.keras import WandbCallback
from glob import iglob
import shutil
import tensorflow as tf
import numpy as np
import requests

# 1. Start a W&B run
wandb.init(project='gnn-challenge-bnn-2021', entity='mfarreras')

wandb_config = wandb.config
#wandb.run.name = subprocess.check_output(['bash','-c', 'echo $NAME']) 

# In case you want to disable GPU execution uncomment this line
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def telegram_bot_sendtext(bot_message):

   bot_token = '5098877072:AAG_D1ww2iOTkGdOYMmOVCAqqqFuPoQbgEY'
   bot_chatID = '8611733'
   send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

   response = requests.get(send_text)

   return response.json()

def transformation(x, y, minmax_values):
    """Apply a transformation over all the samples included in the dataset.
           Args:
               x (dict): predictor variables.
               y (array): target variable.
           Returns:
               x,y: The modified predictor/target variables.
    """
    """tf.print(tf.reduce_min(x['capacity']))
    tf.print(tf.reduce_max(x['capacity']))
    tf.print(tf.reduce_min(x['traffic']))
    tf.print(tf.reduce_max(x['traffic']))
    tf.print(tf.reduce_min(x['packets']))
    tf.print(tf.reduce_max(x['packets']))
    print("TRANSFORMATION")
    x['capacity'] = tf.math.divide(
                    tf.math.subtract(x['capacity'],tf.reduce_min(x['capacity'])),
                    tf.math.subtract(tf.reduce_max(x['capacity']),tf.reduce_min(x['capacity']))
                    )
    x['traffic'] = tf.math.divide(
                    tf.math.subtract(x['traffic'],tf.reduce_min(x['traffic'])),
                    tf.math.subtract(tf.reduce_max(x['traffic']),tf.reduce_min(x['traffic']))
                    )
    x['packets'] = tf.math.divide(
                    tf.math.subtract(x['packets'],tf.reduce_min(x['packets'])),
                    tf.math.subtract(tf.reduce_max(x['packets']),tf.reduce_min(x['packets']))
                    )
    #(x['packets']-tf.reduce_min(x['packets']))/(tf.reduce_max(x['packets'])-tf.reduce_min(x['packets']))
    st_minmax = {
        'traffic': [30.787, 2048.23],
        'packets': [0.032895, 2.03633],
        'capacity':[10000, 100000]
    }
    
    for k,v in st_minmax.items():
        denom = v[1] - v[0]
        x[k] = tf.math.divide(tf.math.subtract(x[k], v[0]), denom)"""
    """for k,v in minmax_values.items():
        denom = v[1] - v[0]
        x[k] = tf.math.divide(tf.math.subtract(x[k], v[0]), denom)
"""
    x['traffic'] = (x['traffic']-minmax_values['traffic'][0])/(minmax_values['traffic'][1]-minmax_values['traffic'][0])
    x['packets'] = (x['packets']-minmax_values['packets'][0])/(minmax_values['packets'][1]-minmax_values['packets'][0])
    
    return x, y
    """x['capacity'] = (x['capacity']-tf.reduce_min(x['capacity']))/(tf.reduce_max(x['capacity'])-tf.reduce_min(x['capacity']))
    x['traffic'] = (x['traffic']-tf.reduce_min(x['traffic']))/(tf.reduce_max(x['traffic'])-tf.reduce_min(x['traffic']))
    x['packets'] = (x['packets']-tf.reduce_min(x['packets']))/(tf.reduce_max(x['packets'])-tf.reduce_min(x['packets']))
    #x['pathLength'] = (x['pathLength']-tf.reduce_min(x['pathLength']))/(tf.reduce_max(x['pathLength'])-tf.reduce_min(x['pathLength']))
    return x, y"""


minmax = {
        'traffic': [0, 2048.23],
        'packets': [0, 2.03633]
        #'capacity':[10000, 100000]
}

"""validation_minmax = {
    'traffic': [30.543, 2064.93],
    'packets': [0.03107, 2.03985],
    'capacity':[10000, 2500000]
}

test_minmax = {
    'traffic': [0, 2061.17],
    'packets': [0, 2.0543],
    'capacity':[10000, 2500000]
}"""

#Dictionary with standardization values



    #x['pathLength'] = (x['pathLength']-tf.reduce_min(x['pathLength']))/(tf.reduce_max(x['pathLength'])-tf.reduce_min(x['pathLength']))
    #return x, y

# Read the config file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

# Initialize the datasets
"""
ds_train = input_fn(config['DIRECTORIES']['train'], shuffle=True)
ds_train = ds_train.map(lambda x, y: transformation(x, y, minmax))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()

ds_val = input_fn(config['DIRECTORIES']['val'], shuffle=False)
ds_val = ds_val.map(lambda x, y: transformation(x, y, minmax))
ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)"""
wandb.run.name = "GAIN5_prediction"+wandb_config.folder_predicted

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=float(config['HYPERPARAMETERS']['learning_rate']))

# 2. Save model inputs and hyperparameters

wandb_config.learning_rate = float(config['HYPERPARAMETERS']['learning_rate'])
wandb_config.logs = config['DIRECTORIES']['logs']
wandb_config.val = config['DIRECTORIES']['val']
wandb_config.train = config['DIRECTORIES']['train']
wandb_config.link_state_dim = config['HYPERPARAMETERS']['link_state_dim']
wandb_config.path_state_dim = config['HYPERPARAMETERS']['path_state_dim']
wandb_config.t = config['HYPERPARAMETERS']['t']
wandb_config.readout_units = config['HYPERPARAMETERS']['readout_units']
wandb_config.epochs = config['RUN_CONFIG']['epochs']
wandb_config.steps_per_epoch = config['RUN_CONFIG']['steps_per_epoch']
wandb_config.validation_steps = config['RUN_CONFIG']['validation_steps']

# Define, build and compile the model
model = RouteNetModel_predict(wandb_config)

loss_object = tf.keras.losses.MeanAbsolutePercentageError()


model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=True,
              metrics="MAPE")

# Define the checkpoint directory where the model will be saved
ckpt_dir = config['DIRECTORIES']['logs']#+wandb_config.link_state_dim+wandb_config.path_state_dim+wandb_config.readout_units+wandb_config.t+"norm_0"
latest = tf.train.latest_checkpoint(ckpt_dir)

# Reload the pretrained model in case it exists
if latest is not None:
    #shutil.rmtree(ckpt_dir)
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}-{val_MAPE:.2f}")

# If save_best_only, the program will only save the best model using 'monitor' as metric
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode='min',
    monitor='val_MAPE',
    save_best_only=True,
    save_weights_only=True,
    save_freq='epoch')

#es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_MAPE", patience=20)

logging_callback = WandbCallback(log_evaluation=True)

# This method trains the model saving the model each epoch.
"""model.fit(ds_train,
          epochs=int(config['RUN_CONFIG']['epochs']),
          steps_per_epoch=int(config['RUN_CONFIG']['steps_per_epoch']),
          validation_data=ds_val,
          validation_steps=int(config['RUN_CONFIG']['validation_steps']),
          callbacks=[cp_callback,logging_callback],#, es_callback],
          use_multiprocessing=True)"""

# This method evaluates the trained model and outputs the desired metrics for all the test dataset.
#model.evaluate(ds_test)

#CALL MAPE SCRIPT AND COLLECT RESULT
#WRITE RESULT TO WANDBI and screen
ds_test = input_fn_predict(config['DIRECTORIES']['test']+wandb_config.folder_predicted, shuffle=False)
ds_test = ds_test.map(lambda x, y: transformation(x, y, minmax))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# Generate predictions
pred = model.predict(ds_test, verbose=1)

target = tf.concat([y for x, y in ds_test], axis=0)

loss_object = tf.keras.losses.MeanAbsolutePercentageError()

MAPE_ensemble = loss_object(np.squeeze(target),np.squeeze(pred))

print('MAPE ensemble =', MAPE_ensemble.numpy())

print("MAPE: "+str(MAPE_ensemble.numpy())+" %")

file = open("mape.txt", "w")
file.write(str(MAPE_ensemble.numpy()))
file.close()

test = telegram_bot_sendtext(str(MAPE_ensemble.numpy())+" 0 min normalization "+wandb_config.folder_predicted)
print(test)
wandb.log({"test_MAPE": str(MAPE_ensemble.numpy())})