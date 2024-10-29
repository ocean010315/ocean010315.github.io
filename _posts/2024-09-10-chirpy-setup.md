---
title: Chirpy theme 적용하기
date: 2024-09-10 00:00:00 +09:00
categories: [github.io, chirpy 세팅]
tags: [chirpy, theme]
use_math: true
comments: true
---

### Chirpy 선택 이유
많은 사람들이 사용하는 거라 레퍼런스가 많을 것이라고 생각했는데 <s>개뿔</s> 루비 3.3 버전을 쓰면서 이런 저런 오류가 발생했기 때문에 직접 글을 쓴다.<br>
`chirpy starter`는 절대 권장하지 않는다. 세부 세팅을 변경하기가 너무 번거로움!
`Github Fork` 방식도 나는 별로다.. 에러가 발생해서 검색해보니 이슈에도 떠있었고, 해결이 안 되는 경우도 있었기 때문에 추천하지 않는다.

## **사전 작업**
### Ruby 설치
[Ruby 설치 페이지](https://rubyinstaller.org/downloads/)에 접속해서 Ruby를 설치한다. 나는 작성하는 시점에 최신 버전인 3.3.5-1 (x64)를 설치했다. <br>
설치 후 실행하면 설치 페이지가 뜨는데 가급적 전부 체크하고 **Run 'ridk install' to setup**에도 체크한다. 설치가 끝난 후 터미널 창이 뜨면 **Enter**를 눌러서 자동으로 셋업을 완료한다!

이제 필요한 모듈을 설치해준다.
```bash
$ gem install jekyll
$ gem install bundler
```

cmd나 PowerShell 등 원하는 터미널 창을 켜고 아래와 같이 입력했을 때 버전이 뜨면 설치가 잘 완료된 것이다! 버전은 다를 수 있으므로 토씨 하나까지 같지 않아도 될 것이다!(아마?)
```bash
$ ruby -v
ruby 3.3.5 (2024-09-03 revision ef084cc8f4) [x64-mingw-ucrt]

$ jekyll -v
jekyll 4.3.3

$ bundler -v
Bundler version 2.5.18
```

### npm 설치
node.js가 필요하다.
[node.js 설치 가이드](https://velog.io/@ljs923/Node.js-%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C-%EB%B0%8F-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0)는 이 링크를 참고하도록 하자.<br>
아니면 [node.js 설치 페이지](https://nodejs.org/en)로 가서 LTS 버전을 설치한다. 왜냐하면 LTS가 가장 안정적이니까..

그리고 npm도 버전 확인을 함으로써 설치 확인하기.
```bash
$ npm -v
10.8.2
```

Chirpy 공식 설명서를 보면 make도 필요하다고 해서 나는 GNU make까지도 설치를 했었는데 필요 없는 것 같기도 하고..?
혹시 실행하다가 make를 사용하게 되면 [여기](https://jstar0525.tistory.com/264) 참고.

<br>

> 출처 <br>
[Node.js 다운로드 및 설치하기](https://velog.io/@ljs923/Node.js-%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C-%EB%B0%8F-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0) <br>
[[make] Windows 10에서 make 사용하기](https://jstar0525.tistory.com/264)

## **레포 생성 후 Chirpy 적용하기**
### 레포 생성하고 로컬에 clone
github에서 `{username}.github.io`라는 이름의 레포를 하나 파고 로컬로 clone한다. 모두가 할 수 있는 것이라 믿고 과정은 생략할게요~

### Chirpy 테마 가져오기
[Chirpy 다운로드 페이지](http://jekyllthemes.org/themes/jekyll-theme-chirpy/)에서 Chirpy를 설치하고 압축을 해제한다. 그리고 모든 파일을 긁어와서 그대로 나의 빈 레포에 붙여넣기 한다. <br>
그리고 Linux에서는 `bash tools/init.sh`를 입력하면 알아서 initialize가 수행되지만 Windows에서는 지원하지 않기 때문에 손수 initialize를 수행해야 한다. <br>
- `Gemfile.lock` 삭제
- `_posts` 디렉토리 삭제
- `docs` 디렉토리 삭제
- `.github/workflows/pages-deploy.yaml` 파일을 제외한 나머지 파일 전부 삭제
    * 여기서 `.github/workflows` 하단에 starter 관련 디렉토리 하위에 `pages-deploy.yaml`이 있는데, 이거 빼고 `workflows`에 있는 모든 파일을 삭제하고 `pages-deploy.yaml`을 starter 디렉토리에서 꺼내주면 된다.
    * 그리고 당연히 starter 디렉토리도 지워준다.
- 마지막으로 `Gemfile`에 있는 `wdm` 관련 라인을 지운다. Ruby의 현재 버전에서 해당 모듈을 지원하지 않는다. 나중에 깔으라고 끊임없이 요구하는데 걍 무시하면 됨.

그리고!!!
```bash
$ bundle install
$ npm install
$ npm run build
```
하면 완료. `jekyll serve`를 입력해서 나타나는 로컬 서버로 접속하면 Chirpy가 적용된 github.io를 로컬에서 확인할 수 있다.

근데 로컬에서 실행했다고 github에 push했을 때도 된다고는 안 했다.. <br>
`.gitignore`에서 `# Misc` 하위 목록을 전부 주석 처리한다. 즉, `_saas/dist`와 `assets/js/dist`를 주석 처리해서 변경사항이 반영되게 해야 npm으로 run, build한 내역이 반영된다.

## **Chirpy 커스터마이징**
### `_config.yaml` 설정
주석이 친절하게 달려 있으므로 잘 설정하면 된다. 아니면 여기저기 검색해서 참고하기.

### 블로그 이미지 설정
프로필 이미지는 assets/img에 넣고 관리하기. 나는 avatar.jpg로 넣어놨다. 그리고 다른 이미지들도 여기에 넣어두고 관리하면 된다.<br>

### 블로그 띄우기
config를 다 수정했으면 이제 push해서 원격 저장소에 반영한다. 하지만 나는 앞서 `.github/workflows` 하위에 있는 yml이랑 또 오류를 겪었기 때문에 새롭게 yml 파일을 만들어줬다. <br>
1. 먼저 Github에 접속해서 Settings > pages > Build and deployment에 차례대로 접근한다.
2. Deploy from a branch라고 되어있는 설정을 Github Actions로 바꾸고 configure를 선택한다.
3. 수정 없이 Commit changes...를 누르고 또 Commit changes를 눌러준다. <br>

알아서 빌드될 것! 그리고 로컬로 돌아와 변경된 yml을 가져오기 위해 pull한다.

## 앞으로는 `_posts` 디렉토리에 글 작성하면 된다!

<br><br><br>
> **참고** <br>
[Chirpy 적용하기](https://devpro.kr/posts/Github-%EB%B8%94%EB%A1%9C%EA%B7%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-(4)/) <br>
[jekyll.yml 생성하기](https://ree31206.tistory.com/entry/github-pages-%EB%B8%94%EB%A1%9C%EA%B7%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-%ED%85%8C%EB%A7%88-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0Chirpy)