import os
import tensorflow as tf
import tensorflowjs as tfjs
from transformers import BertTokenizerFast, TFBertForQuestionAnswering, DefaultDataCollator
from datasets import load_dataset
from transformers import create_optimizer
from transformers import TrainingArguments, TFTrainer

def train_and_export_model():
    # 1. Cargar el dataset en español
    dataset = load_dataset("squad_es")

    # 2. Tokenizador y modelo base (entrenable)
    model_name = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = TFBertForQuestionAnswering.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    # 3. Convertir a tf.data.Dataset
    train_dataset = tokenized_datasets["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "token_type_ids"],
        label_cols=["start_positions", "end_positions"],
        shuffle=True,
        batch_size=8,
        collate_fn=DefaultDataCollator()
    )

    # 4. Entrenamiento
    num_train_steps = len(train_dataset) * 3
    optimizer, _ = create_optimizer(init_lr=2e-5, num_train_steps=num_train_steps, num_warmup_steps=0)

    model.compile(optimizer=optimizer)
    model.fit(train_dataset, epochs=3)

    # 5. Exportar modelo a TensorFlow.js
    export_dir = "exported_model"
    os.makedirs(export_dir, exist_ok=True)

    tf.saved_model.save(model, export_dir)

    tfjs.converters.convert_tf_saved_model(
        export_dir,
        output_dir="model_tfjs",
    )

    print("Entrenamiento y exportación completa. Modelo listo en carpeta 'model_tfjs'.")

# Ejecutar si es llamado directamente
if __name__ == "__main__":
    train_and_export_model()
