# 파일명: inference.py

import json
import os
import torch
import torch.nn as nn # <--- nn import 추가
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sentence_transformers import SentenceTransformer, util

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    print("="*50)
    print("Step 3: Inference (Config-based with Projection Head)")
    print("="*50)

    # --- 설정 파일 로드 ---
    config = load_config()
    model_config = config['model']
    path_config = config['path']

    # --- 디바이스 설정 ---
    if config['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['device'])
        
    # --- 모델 및 경로 확인 ---
    model_path = os.path.join(path_config['model_save_dir'], 'zsl_audio_model.pth')
    split_file = path_config['split_file']

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'. Please run train.py first.")
        return
    if not os.path.exists(split_file):
        print(f"Error: Split file not found at '{split_file}'. Please run preprocess.py first.")
        return

    # --- 모델 불러오기 ---
    print(f"Loading fine-tuned models from '{model_path}' to device: {device}...")
    audio_model = Wav2Vec2Model.from_pretrained(model_config['audio_model_name'])
    text_model = SentenceTransformer(model_config['text_model_name'])
    audio_processor = Wav2Vec2Processor.from_pretrained(model_config['audio_model_name'])
    
    # --- [수정 1] 프로젝션 헤드 구조 정의 ---
    audio_projection_head = nn.Linear(768, 384)

    checkpoint = torch.load(model_path, map_location=device)
    audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
    text_model.load_state_dict(checkpoint['text_model_state_dict'])
    # --- [수정 2] 프로젝션 헤드 가중치 불러오기 ---
    audio_projection_head.load_state_dict(checkpoint['audio_projection_head_state_dict'])
    
    audio_model.to(device).eval()
    text_model.to(device).eval()
    audio_projection_head.to(device).eval() # 프로젝션 헤드도 평가 모드로

    # --- 데이터 준비 ---
    print("Loading and filtering dataset for zero-shot testing...")
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    unseen_classes = split_info['unseen_classes']

    unseen_dataset = load_dataset("ashraq/esc50", split="train").filter(
        lambda x: x['category'] in unseen_classes
    )

    # --- 판별(Inference) 실행 ---
    correct_predictions = 0
    total_samples = len(unseen_dataset)
    
    text_descriptions = [f"the sound of a {label.replace('_', ' ')}" for label in unseen_classes]
    with torch.no_grad():
        unseen_text_embeddings = torch.from_numpy(text_model.encode(text_descriptions)).to(device)

    print(f"\nStarting inference on {total_samples} unseen samples...")
    for i, sample in enumerate(unseen_dataset):
        raw_audio = sample['audio']['array']
        true_label = sample['category']
        
        resampled_audio = librosa.resample(raw_audio, orig_sr=44100, target_sr=16000)
        audio_inputs = audio_processor(resampled_audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            audio_embedding = audio_model(**audio_inputs).last_hidden_state.mean(dim=1)
            # --- [수정 3] 추론 시에도 프로젝션 헤드 통과 ---
            projected_audio_embedding = audio_projection_head(audio_embedding)

        # --- [수정 4] 프로젝션된 벡터로 유사도 계산 ---
        cosine_scores = util.cos_sim(projected_audio_embedding, unseen_text_embeddings)
        predicted_index = torch.argmax(cosine_scores)
        predicted_label = unseen_classes[predicted_index]

        result_marker = "✅" if predicted_label == true_label else "❌"
        if predicted_label == true_label:
            correct_predictions += 1
        
        print(f"Sample {i+1:>3}/{total_samples}: True: {true_label:<20} | Pred: {predicted_label:<20} | {result_marker}")

    # --- 최종 성능 평가 ---
    accuracy = (correct_predictions / total_samples) * 100
    print("\n" + "="*50)
    print("--- Zero-Shot Classification Result ---")
    print(f"Total Unseen Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()