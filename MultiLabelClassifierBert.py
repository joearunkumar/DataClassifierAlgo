import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from transformers import TFBertModel, BertConfig, BertTokenizerFast

# -----------------------------------------------------------------#
# Import, clean, and process data

# Import data from csv
data = pd.read_csv('filtered_data.csv')

# Select required columns
data = data[['Comments', 'Classification']]

# Set your model output as categorical and save in new label col
data['Classification_label'] = pd.Categorical(data['Classification'])

# Transform your output to numeric
data['Classification_num'] = data['Classification_label'].cat.codes

dict = {'label': data['Classification_num'], 'category': data['Classification']}
df = pd.DataFrame(dict)
df.to_csv('Bert_Labels.csv')

data['Classification'] = data['Classification_num']

# Split into train and test - stratify over Issue
data, data_test = train_test_split(data, test_size=0.2, stratify=data[['Classification']])

# -----------------------------------------------------------------#
# Setup BERT

# Name of the BERT model to use
model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 250

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config=config)


# -----------------------------------------------------------------#
# Build the model
# TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model

# Load the MainLayer
bert = transformer_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
# attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
# inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Then build your model output
classification = Dense(units=len(data.Classification_label.value_counts()),
                       kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='classification')(
    pooled_output)
outputs = {'classification': classification}

# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')

# Model summary
model.summary()

# -----------------------------------------------------------------#
# Train the model

# Set an optimizer
optimizer = Adam(
    # learning_rate=5e-05,
    learning_rate=1e-5,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = {'classification': CategoricalCrossentropy(from_logits=True)}
metric = {'classification': CategoricalAccuracy('accuracy')}

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metric)

# Ready output data for the model
y_classification = to_categorical(data['Classification'])

# Tokenize the input (takes some time)
x = tokenizer(
    text=data['Comments'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True)

# Fit the model
history = model.fit(
    # x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
    x={'input_ids': x['input_ids']},
    y={'classification': y_classification},
    validation_split=0.2,
    batch_size=32,
    epochs=15)

# -----------------------------------------------------------------#
# Save model
model.save("Bert_Classification_Model")

# -----------------------------------------------------------------#
# Evaluate Model

# Ready test data
test_y_issue = to_categorical(data_test['Classification'])
test_x = tokenizer(
    text=data_test['Comments'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=False,
    verbose=True)

# Run evaluation
model_eval = model.evaluate(
    x={'input_ids': test_x['input_ids']},
    y={'classification': test_y_issue}
)
