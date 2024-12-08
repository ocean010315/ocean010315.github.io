---
title: Django와 MySQL 연동하기 - 2
date: 2024-12-08 00:00:00 +09:00
categories: [개발 일지, Django]
tags: [Django, MySQL, Backend]
use_math: true
---

이전 글에서는 Django 프로젝트 자체와 MySQL을 연동해서 디폴트 테이블들을 생성하는 것까지 수행했다.  
이번에는 사용자 정보를 저장하기 위한 User 앱을 생성하고 관련 테이블을 생성하는 법을 알아볼 것이다.  

### App 생성  
1. Django 프로젝트 내에 회원가입, 로그인, 마이페이지 등에 활용할 User 앱을 생성한다.  
    ```bash
    python manage.py startapp User
    ```

2. 프로젝트 폴더 내 `User`이라는 이름의 디렉토리가 생성되었을 것이다. 이제 `settings.py`에 생성한 앱을 등록한다.  
1번과 2번의 순서가 바뀌면 migrate 할 때 앱을 찾을 수 없다는 오류가 뜨게 되므로 주의하도록 한다.
    ```python
    # settings.py
    INSTALLED_APPS = [
        ...,
        'User',
    ]
    ```

### 테이블 생성, migrate 수행
1. `User/models.py`에 생성할 테이블의 구조를 설계한다. 난 단순한 CRUD 복습을 위한 거라 이름, 아이디 대용 이메일, 비밀번호에 해당하는 필드만 만들었다.  
    ```python
    from django.db import models


    class User(models.Model):
        ID = models.AutoField(primary_key=True)
        name = models.CharField(max_length=50)
        email = models.EmailField(unique=True)
        password = models.CharField(max_length=100)

        class Meta:
            managed = True
            db_table = 'user'
    ```

    - `AutoField` 데이터 삽입 시 1부터 시작해서 자동으로 오름차순으로 값이 지정되는 정수 필드이다.
    - `CharField` 글자를 삽입하는 필드이고 최대 길이를 지정해야 한다. 길이 제한이 없는 텍스트 필드는 `TextField`를 사용한다.
    - `EmailField` 이름처럼 email 주소를 저장하는 필드이다.

    > **`Meta` 클래스**  
    Django의 `models.Model`라는 추상 클래스를 상속받아 모델을 구현할 경우 `Meta` 클래스를 내부에 선언함으로써 메타 데이터 정보를 부여할 수 있다. 정렬 방법, 테이블의 이름 등을 지정할 수 있다. 개별 옵션은 [여기](https://docs.djangoproject.com/en/5.1/ref/models/options/)에서 확인하기!

2. 이제 migrate를 한다.
    ```bash
    python manage.py makemigrations User
    python manage.py migrate User
    ```
    이 때 [이전 포스트]("https://ocean010315.github.io/posts/django-mysql-1/")와 같이 프로젝트와 MySQL을 미리 연동하지 않았다면 `User`를 빼고 전체적으로 연동이 되도록 한다.