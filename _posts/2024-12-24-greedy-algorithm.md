---
title: Greedy Algorithm
date: 2024-12-24 00:00:00 +09:00
categories: [스터디, 알고리즘]
tags: [Greedy]
use_math: true
---

### Greedy Algorithm
- 매 순간 최고의 선택지를 따라가는 알고리즘  
- 최적의 해를 보장하지 않음  
- 따라서 코딩 테스트의 경우, 탐욕법으로 도달한 해가 최적의 해가 되는 문제를 출제  

## 백준 문제 풀기
> 난이도별로 구성  

### 10610번 30
**`Link`** [https://www.acmicpc.net/problem/10610](https://www.acmicpc.net/problem/10610)  
**`Problem`**  
1. N: 양의 정수
2. N을 구성하는 각 자리의 수를 활용하여 새로운 숫자 생성
3. 해당 숫자는 30의 배수 중 최댓값
4. 30의 배수를 만들 수 없다면 -1 출력

**`Code`**  
```python
N = [int(i) for i in input()]
if sum(N) % 3 == 0 & 0 in N:
    N = [str(n) for n in N]
    print(''.join(sorted(N, reverse=True)))
else:
    print(-1)
```

**`Solution`**  
30의 배수는 3의 배수이면서 10의 배수이다. 3의 배수는 각 자릿수의 합이 3의 배수라는 특징이 있기 때문에 N의 각 자릿수의 합이 3인지 확인한다. 또한 N에 0이 포함되어 있지 않다면 새로 만든 숫자가 10의 배수가 될 수 없기 때문에 바로 -1을 출력한다.  
3의 배수와 10의 배수의 조건을 통과했다면 각 자릿수를 내림차순으로 정렬하여 출력하기만 하면 된다. 0은 한 자리 자연수 중 가장 작은 수이기 때문에 자동으로 일의 자리에 배치될 것이다.  

### 11047번 동전 0
**`Link`** [https://www.acmicpc.net/problem/11047](https://www.acmicpc.net/problem/11047)  
**`Problem`**  
1. N: 가지고 있는 동전의 종류의 개수, K: 만들어야 하는 값
2. 1번째 줄부터 N+1번째 줄까지 동전의 종류가 오름차순으로 주어지며, 다음 동전은 반드시 이전 동전의 배수
3. K값에 도달하기 위해 사용할 수 있는 동전의 개수 중 최솟값

**`Code`**  
```python
N, K = [int(i) for i in input().split()]
coins = [int(input()) for _ in range(N)]
coins.sort(reverse=True)
cnt = 0
for c in coins:
    if c <= K:
        cnt += K // c
        K %= c
        if K == 0: break
print(cnt)
```

**`Solution`**  
흔히 아는 거스름돈 문제와 비슷한데, 가장 값이 높은 동전에서부터 하나씩 계산하면 된다. 따라서 동전을 내림차순으로 재정렬 했으며, 몫과 나머지를 구해가는 방식으로 구현했다.  
큰 동전을 최대한 많이 사용하면 적은 개수만으로도 K값에 더 빠르게 도달이 가능하다. 또한 큰 동전은 작은 동전의 배수이기 때문에 작은 동전으로 반드시 큰 동전의 값을 만들 수 있다. 따라서 큰 동전부터 사용해도 남은 값들을 작은 동전으로 채울 수 있기 때문에 Greedy Algorithm으로 해결 가능하다.  
Ex. 1: [1, 5, 10]으로 13을 만들 때 10원 1개, 1원 3개로 총 4개가 필요하며, 이게 최적의 값임
Ex. 2: [1, 3, 4]로 6을 만들 때 4원을 먼저 사용하면 4원 1개, 1원 2개로 총 3개가 필요하지만, 최적의 값은 3원 2개를 사용하는 것

### 2875번 대회 or 인턴
**`Link`** [https://www.acmicpc.net/problem/2875](https://www.acmicpc.net/problem/2875)  
**`Problem`**  
1. N: 여학생의 수, M: 남학생의 수, K: 인턴쉽에 참여해야 하는 최소 인원의 수  
2. 전체 인원은 팀 대회, 인턴쉽에 나누어 참여
3. 한 팀은 여학생 2명, 남학생 1명으로 구성되어야 하고, 전체 인원 중 최소한 K명은 인턴쉽에 참여  
4. 만들 수 있는 가장 많은 팀의 수 구하기

**`Code`**  
```python
N, M, K = [int(i) for i in input().split()]
teams = 0
while N+M > K: 
    N -= 2
    M -= 1
    if N+M < K or N<0 or M<0:
        break
    else:
        teams += 1
print(teams)
```

**`Solution`**  
한 팀씩 개설 후 남은 인원을 세는 방식이다.  
처음에는 조건문에 `N+M <= K`라고 지정해서 틀렸는데, 팀을 개설하고 남은 인원이 K여도 팀은 개설될 수 있기 때문에 등호를 빼고 `N+M < K`라고 지정해야 한다.

### 1931번 회의실 배정
**`Link`** [https://www.acmicpc.net/problem/1931](https://www.acmicpc.net/problem/1931)  
**`Problem`**  
1. N: 회의의 개수
2. N개의 회의에 대하여 회의실 사용표를 제작
3. 할 수 있는 회의의 최대 개수
4. N개의 줄에 걸쳐 각 회의 I의 시작 시간과 종료 시간이 주어짐
5. 회의는 겹칠 수 없고, 한 번 시작하면 종료할 수 없음

**`Code`**  
```python
N = int(input())
I = [[int(i) for i in input().split()] for _ in range(N)]
I.sort(key=lambda x:(x[1], x[0]))
now = 0
cnt = 0
for i in I:
    if i[0] >= now:
        cnt += 1
        now = i[1]
print(cnt)
```

**`Solution`**  
종료 시간이 빠른 순서대로 정렬하고, 종료 시간이 같다면 시작 시간이 빠른 순서대로 정렬한다. `now`라는 변수를 활용해서 현재 시간 이후의 회의만 고려하도록 한다.  
종료 시간이 빠른 순서대로 정렬하면 해당 회의 이후에 다른 회의를 선택할 가능성들이 보다 커진다. 또한 `now` 변수를 활용하여 회의 시간이 겹치지 않기 때문에 Greedy Algorithm으로 분류된다.  

### 1783번 병든 나이트
**`Link`** [https://www.acmicpc.net/problem/1783](https://www.acmicpc.net/problem/1783)  
**`Problem`**  
1. N: 체스판의 세로 길이, M: 체스판의 가로 길이
2. 병든 나이트가 움직일 수 있는 방법은 아래 네 가지 방법 뿐  
  a. 2칸 위, 1칸 오른쪽  
  b. 1칸 위, 2칸 오른쪽  
  c. 1칸 아래, 2칸 오른쪽  
  d. 2칸 아래, 1칸 오른쪽  
3. 시작점은 체스판의 왼쪽 아래
4. 체스판 내에서 한 번에 이동할 수 있는 최댓값
    - 체스판 내에서 방문할 수 있는 칸의 최대 개수가 아닌, 어느 한 지점에 도달했을 때 해당 지점이 가장 많은 횟수를 거쳐 도달한 위치여야 함
5. 4번 이상 움직인다면 네 방법을 반드시 최소 한 번씩 사용해야 하며, 그렇지 않다면 움직임의 제약이 없음

**`Code`**  
```python 
N, M = [int(i) for i in input().split()]
if N == 1: print(1)
elif N == 2: print(min((M+1)//2, 4))
elif M < 7: print(min(M, 4))
else: print(M-2)
```

**`Solution`**  
처음엔 방문할 수 있는 모든 칸의 수를 세는 건 줄 알고 DFS, BFS 방식으로 접근했다.. 문제를 잘못 이해했다..😂  
가로 길이가 7이고 세로 길이가 3이면 a, b, c, d 모든 방법을 한 번씩 사용할 수 있다. 따라서 이 값을 기준으로 한다.  
또한 a, b, c, d 모든 방법이 오른쪽으로는 이동하지만 세로 방향으로는 올라갔다가 내려올 수 있다는 점을 기억한다.

세로 길이가 1일 때
- 네 개의 방법에 위, 아래 움직임이 포함되어 있기 때문에 시작점 외에 갈 수 있는 곳이 없다.  

세로 길이가 2일 때
- b, c 방법밖에 사용할 수 없다. 이 때 오른쪽으로 두 칸씩 움직이기 때문에 가로 길이가 1, 2일 때는 **1**(시작점), 3, 4일 때는 **2**와 같은 패턴을 가진다.
- 이동 횟수가 4를 초과하면 a, b 방법은 포함되지 않았기 때문에 최종 출력은 **4**로 제한한다.
- `min` 함수를 사용하여 위 두 로직을 표현한다.

가로 길이가 7 미만일 때(세로 길이는 3 이상)
- 세로 길이가 3 이상이기 때문에 a, b, c, d에 제약이 없으며 오른쪽으로 한 칸씩만 움직이는 a, d 방식을 사용하는 것이 유리하다.
- 가로 길이가 곧 이동 횟수가 되며, 가로 길이가 7 미만이라 네 개의 방법을 모두 사용할 수 없기 때문에 `min` 함수를 사용한다.  

가로 길이 7 이상, 세로 길이 3 이상
- 자유롭게 움직일 수 있는데, 오른쪽으로 2칸씩 움직여야 하는 b, c를 최소 한 번씩 포함해야 하기 때문에 `가로 길이 - 2`를 출력한다.


> **참고**  
[(이코테 2021 강의 몰아보기) 2. 그리디 & 구현](https://www.youtube.com/watch?v=2zjoKjt97vQ&list=PLRx0vPvlEmdAghTr5mXQxGpHjWqSz0dgC&index=2)  