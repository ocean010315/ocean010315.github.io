---
title: Django의 Serializer
date: 2024-12-08 00:00:00 +09:00
categories: [개발 일지, Django]
tags: [Django, Serializer, Backend]
use_math: true
---

### Serializer란?
Serilizer는 쿼리셋, 모델 인스턴스와 같은 복잡한 데이터를 파이썬 데이터 타입으로 변환하여 JSON, XML로 렌더링 하기 편하게 하는 방법이다.  
자세한 설명은 [Django REST framework 문서](https://www.django-rest-framework.org/api-guide/serializers/#modelserializer) 참고하기!

### Serializer 적용하기
1. `rest_framework` 모듈을 사용할 것이기 때문에 아래 명령어로 설치 진행
    ```bash
    pip install djangorestframework
    ```

2. `settings.py`에 앱 추가
    ```python
    # settings.py
    INSTALLED_APPS = [
        ...,
        'rest_framework',
    ]
    ```

3. 앞서 생성한 `User` 앱의 디렉토리 내 `serializers.py` 생성
    ```python
    from rest_framework import serializers

    from models import User


    class UserSerializer(serializers.ModelSerializer):
        class Meta:
            model = User
            fields = '__all__'

        def create(self, validated_data):
            user = User(
                name=validated_data['name'],
                email=validated_data['email'],
                password=validated_data['password']
            )

            return user
    ```