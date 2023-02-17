                    # This is the additional code
                    pred = outputs.cpu().argmax(-1)
                    data['sample'].extend(tokenized_dataset["text"][i:i+step])
                    data['true_label'].extend(tokenized_dataset["one_hot_label"][i:i+step][:, epm_idx])
                    data['prediction_layer_'+str(epm_idx)] = pred.tolist()



import pandas as pd

class MDL_probe_trainer(Trainer):
    def __init__(self, language_model, dataset_handler: Dataset_handler, verbose=True, device='cuda', pool_method="attn",
                 start_eval=False, normalize_layers=False, early_stopping_patience=2, output_csv_path=None):
        self.output_csv_path = output_csv_path
        # ...

        self.df = pd.DataFrame(columns=["sample", "true_label", "prediction"])

    def train(self, batch_size, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            preds = []

            # ...

            for i in tqdm(range(0, train_len, batch_size), desc=f"[Epoch {epoch + 1}/{epochs}]"):
                # ...

                for epm_idx, edge_probe_model in enumerate(self.edge_probe_models):
                    # ...

                    # Get the predictions
                    preds_batch = edge_probe_model(spans_torch_dict)
                    preds_batch = preds_batch.argmax(dim=1).detach().cpu().numpy()

                    # Append the predictions to the list
                    preds.append(preds_batch)

                    # Store the samples, true labels, and predictions in the DataFrame
                    for j in range(len(preds_batch)):
                        sample = self.dataset_handler.decode_sample(train_dataset[i + j])
                        true_label = self.dataset_handler.labels_list[labels[j].item()]
                        prediction = self.dataset_handler.labels_list[preds_batch[j]]
                        self.df = self.df.append({"sample": sample, "true_label": true_label, "prediction": prediction},
                                                 ignore_index=True)

            # Concatenate the predictions and compute the metrics
            preds = np.concatenate(preds)
            # ...

        if self.output_csv_path is not None:
            self.df.to_csv(self.output_csv_path, index=False)
