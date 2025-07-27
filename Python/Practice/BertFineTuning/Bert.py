from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def encode_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
    )


texts = [example["text"] for example in train_data]

if __name__ == "__main__":
    train_encodings = train_data.map(
        encode_batch, batched=True, num_proc=4, remove_columns=["text"]
    ).with_format("torch")
    test_encodings = test_data.map(
        encode_batch, batched=True, num_proc=4, remove_columns=["text"]
    ).with_format("torch")

    bert = BertModel.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(train_encodings, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_encodings, batch_size=8)

    class BertForSentimentAnalysis(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.dropout = nn.Dropout(0.1)  # 防止过拟合
            self.classifier = nn.Linear(768, 2)  # 768→2（2个类别）

        def forward(self, input_ids, attention_mask):
            # 获取BERT输出，outputs[0]为所有token的隐藏状态，outputs[1]为[CLS]的隐藏状态（部分版本需手动提取）
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cls_hidden = outputs[1]  # [CLS]的隐藏状态，形状为(batch_size, 768)
            cls_hidden = self.dropout(cls_hidden)  #  dropout层减少过拟合
            logits = self.classifier(cls_hidden)
            return logits

    model = BertForSentimentAnalysis().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["label"].to(device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
    torch.save(model.state_dict(), "bert_sentiment_model.pt")

    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()

    accuracy = total_correct / len(test_encodings)
    print(f"Test Accuracy: {accuracy:.4f}")
