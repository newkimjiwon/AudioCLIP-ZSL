# 파일명: preprocess.py

import json
import os
from datasets import load_dataset

def main():
    """
    ESC-50 데이터셋의 클래스를 '학습용(seen)'과 '테스트용(unseen)'으로 분리하고,
    그 목록을 JSON 파일로 저장합니다.
    """
    print("="*50)
    print("Step 1: Preprocessing - Defining data splits")
    print("="*50)

    # 결과물을 저장할 'meta' 폴더 생성
    os.makedirs('meta', exist_ok=True)
    
    print("Loading ESC-50 dataset to define splits...")
    try:
        full_dataset = load_dataset("ashraq/esc50", split="train")
    except Exception as e:
        print(f"Failed to load dataset. Check your internet connection or Hugging Face status. Error: {e}")
        return
        
    class_names = sorted(list(set(full_dataset['category'])))
    
    # 1. '미지(Unseen)'의 클래스 직접 지정
    # 이 클래스들은 학습에 전혀 사용되지 않으며, 최종 Zero-shot 성능 평가에만 사용됩니다.
    unseen_classes = [
        'chainsaw', 
        'clock_tick', 
        'crying_baby', 
        'helicopter', 
        'sneezing',
        'rooster',
        'sea_waves',
        'keyboard_typing',
        'wind',
        'drinking_sipping'
    ]

    # 2. '학습용(Seen)' 클래스 자동 정의
    seen_classes = [c for c in class_names if c not in unseen_classes]
    
    print(f"\nTotal {len(class_names)} classes found.")
    print(f"Defined {len(seen_classes)} classes for training (Seen).")
    print(f"Defined {len(unseen_classes)} classes for zero-shot testing (Unseen).")

    # 3. 분리된 클래스 목록을 JSON 파일로 저장
    split_info = {
        'seen_classes': seen_classes,
        'unseen_classes': unseen_classes
    }
    
    output_path = 'meta/data_split.json'
    with open(output_path, 'w') as f:
        json.dump(split_info, f, indent=4)
        
    print(f"\nSuccessfully created '{output_path}'.")
    print("Preprocessing complete.")

if __name__ == '__main__':
    main()