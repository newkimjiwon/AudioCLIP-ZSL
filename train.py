# 파일명: train.py (Projection Head 추가 버전)

import json
import os
import torch
import torch.nn as nn # <--- nn import 추가
from torch.utils.data import DataLoader
import librosa
from tqdm import tqdm
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AdamW
from sentence_transformers import SentenceTransformer

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    print("="*50)
    print("Step 2: Training (Config-based with Projection Head)")
    print("="*50)

    # --- 설정 파일 로드 ---
    config = load_config()
    train_config = config['train']
    model_config = config['model']
    path_config = config['path']

    # --- 디바이스 설정 ---
    if config['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['device'])
    
    os.makedirs(path_config['model_save_dir'], exist_ok=True)

    # --- 모델 불러오기 ---
    print(f"Loading pre-trained models to device: {device}...")
    audio_processor = Wav2Vec2Processor.from_pretrained(model_config['audio_model_name'])
    audio_model = Wav2Vec2Model.from_pretrained(model_config['audio_model_name']).to(device)
    text_model = SentenceTransformer(model_config['text_model_name']).to(device)
    
    # --- [수정 1] 프로젝션 헤드 정의 ---
    # 768차원(오디오) -> 384차원(텍스트)으로 변환하는 Linear 레이어
    audio_projection_head = nn.Linear(768, 384).to(device)

    audio_model.feature_extractor.requires_grad_(False)

    # --- 데이터 준비 ---
    print("Loading and filtering dataset for training...")
    with open(path_config['split_file'], 'r') as f:
        split_info = json.load(f)
    seen_classes = split_info['seen_classes']

    train_dataset = load_dataset("ashraq/esc50", split="train").filter(
        lambda x: x['category'] in seen_classes
    ).shuffle(seed=42)
    
    def collate_fn(batch):
        raw_audios = [item['audio']['array'] for item in batch]
        labels = [item['category'] for item in batch]
        return {'audio': raw_audios, 'category': labels}

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config['batch_size'],
        collate_fn=collate_fn
    )
    
    print(f"Training with {len(train_dataset)} samples from {len(seen_classes)} seen classes.")

    # --- [수정 2] 옵티마이저에 프로젝션 헤드의 파라미터 추가 ---
    optimizer = AdamW(
        list(audio_model.parameters()) + list(text_model.parameters()) + list(audio_projection_head.parameters()), 
        lr=train_config['learning_rate']
    )
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.8)

    # --- 학습 루프 ---
    print(f"Starting training for {train_config['epochs']} epochs...")
    audio_model.train()
    text_model.train()
    audio_projection_head.train() # 프로젝션 헤드도 학습 모드로

    for epoch in range(train_config['epochs']):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']}")
        
        for batch in pbar:
            raw_audios = batch['audio']
            labels = batch['category']
            
            resampled_audios = [librosa.resample(a, orig_sr=44100, target_sr=16000) for a in raw_audios]
            audio_inputs = audio_processor(resampled_audios, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            
            text_descriptions = [f"the sound of a {label.replace('_', ' ')}" for label in labels]

            audio_outputs = audio_model(**audio_inputs).last_hidden_state.mean(dim=1)
            text_outputs = torch.from_numpy(text_model.encode(text_descriptions)).to(device)

            # --- [수정 3] 오디오 벡터를 프로젝션 헤드에 통과 ---
            projected_audio_outputs = audio_projection_head(audio_outputs)
            
            # --- [수정 4] 프로젝션된 벡터로 손실 계산 ---
            target = torch.ones(projected_audio_outputs.size(0)).to(device)
            loss = loss_fn(projected_audio_outputs, text_outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # --- [수정 5] 모델 저장 시 프로젝션 헤드 가중치도 함께 저장 ---
    model_save_path = os.path.join(path_config['model_save_dir'], 'zsl_audio_model.pth')
    print(f"\nTraining complete. Saving fine-tuned model to '{model_save_path}'...")
    torch.save({
        'audio_model_state_dict': audio_model.state_dict(),
        'text_model_state_dict': text_model.state_dict(),
        'audio_projection_head_state_dict': audio_projection_head.state_dict(),
    }, model_save_path)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()