---
title: Django의 Serializer
date: 2024-12-09 00:00:00 +09:00
categories: [개발 일지, Django]
tags: [Django, Serializer, Backend]
use_math: true
---

### Serializer란?
Serilizer는 쿼리셋, 모델 인스턴스와 같은 복잡한 데이터를 파이썬 데이터 타입으로 변환하여 JSON, XML로 렌더링 하기 편하게 하는 방법이다.  
나는 단순한 구조로 진행할 것이기 때문에 보다 자세한 설명과 사용법은 [Django REST framework 문서](https://www.django-rest-framework.org/api-guide/serializers/#modelserializer) 참고하기!

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
여기서 사용하는 serializer는 생성한 model과 필드를 공유하기 때문에 내가 생성한 모델 구조가 궁금하다면 [여기](https://ocean010315.github.io/posts/django-mysql-2/)에서 확인하면 된다.  

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

    - `fields`: serializer에 포함할 데이터 필드를 지정한다. 테이블에 삽입하고 꺼내올 때는 주로 모든 필드를 사용하기 때문에 `__all__`로 지정했으며, 특별히 일부 필드만 사용하고 싶으면 `fields = ('name', 'email')`과 같은 형태를 사용한다.
    - `read_only_fields`: 수정하거나 직접 값을 삽입하지 않고 읽기만 하도록 하는 필드를 지정한다. 테이블을 설계할 때 `editable=False`, `AutoField`로 지정한 필드들은 자동으로 `read_only_fields`로 지정된다.  
    - `create`: 테이블을 생성할 때 사용하는 함수이다. 원래 `serializers.Serializer`를 사용하면 crud 함수마다 직접 내용을 작성해서 커스텀하지만, `serializers.ModelSerializer`는 이 모든 과정을 이미 정의된 함수들로 압축할 수 있게 해준다.

4. views.py 작성
    ```python
    # views.py
    