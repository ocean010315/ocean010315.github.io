---
title: Lecture 3. Planning by Dynamic Programming
date: 2025-09-07 00:00:00 +09:00
categories: [RL, David Silver]
tags: [RL]
use_math: true
---

> 이 포스팅은 David Silver의 RL 강좌를 기반으로 작성되었습니다.  
- [강의 링크](https://www.youtube.com/watch?v=Nd1-UUMVfz4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=3), [강의 자료 링크](https://davidstarsilver.wordpress.com/teaching/)  
- 이미지 출처: David Silver, RL Course (CC-BY-NC 4.0)

## Introduction
**[ Dynamic Programming이란? ]**  
```복잡한 문제를 단순한 subproblem으로 쪼개어 해결하는 방법```  

**[ Requirements for DP ]**  
- 두 가지 요소를 충족해야 함
  - 최적의 해결방법은 subproblem들로 쪼개질 수 있어야 함
  - 쪼개진 subproblem들에 겹치는 부분이 있어 caching이 가능해야 함
- MDP는 두 가지 요소를 모두 충족
  - Bellman Equation을 통해 반복적으로 문제를 쪼갤 수 있음
  - value fn.은 저장되고, 재사용 가능

**[ Planning by DP ]**  
- DP는 MDP의 모든 상황을 알고 있음을 가정
  - model-free: env가 어떤지 모르며, 완전한 정보를 얻을 수 없음 → DP 적용 X
  - model-based: env에 대한 model이 있어 모든 상황을 아는 경우 → DP 적용 O

`for Prediction`  
- Input: MDP $<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma>$ & policy $\pi$ or MRP $<\mathcal{S}, \mathcal{P}^\pi, \mathcal{R}^\pi, \gamma>$  
- Output: value fn. $v_\pi$ (즉, 목표가 value fn.을 예측하는 것)

`for Control`  
- Input: MDP $<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma>$
- Output: 최적의 value fn. $v_*$ 와 최적의 policy $\pi_*$  

DP는 RL에서 뿐만 아니라 다양한 문제 해결에 사용되지만, 본 강의와는 관련 없기 때문에 생략!

## Policy Evaluation
### Iterative Policy Evaluation
`Problem`  
policy $\pi$ 가 주어졌을 때, 해당 policyd의 reward를 계산하는 것

`Solution`  
반복적으로 Bellman Expectation을 적용 ($v_1 \to v_2 \to ... \to v_\pi$)  
이때, synchronous backups(caching)를 활용 (asynchronous backups는 아직 고려하지 않음)
- 각 iteration step $k+1$ 일 때,
- 모든 state $s \in S$ 에 대해서,
- $v_k(s')$ 를 활용해서 $v_{k+1}(s)$ 를 업데이트 (이때, $s'$ 은 $s$ 로부터 도달 가능한 다음 state)

$v_\pi$ 로 반드시 수렴하는 것을 본 강의에서 증명

<div align="center">
<img src="../assets/img/250907_rl3/iterative_policy_eval.png" width="400">
</div>

state $s$ 에서 action $a$ 수행 → 다음 모든 state $s'$ 들의 값을 사용해서 현재 state의 value fn. $v_{k+1}(s)$ 를 정확히 예측  
- $\pi(a \mid s)$: 현재 state $s$ 에서 수행할 수 있는 action $a$ 에 대한 policy
- $\mathcal{R}_s^a$: 현재 state $s$ 에서 action $a$ 를 수행할 때 받는 immediate reward
- $\gamma$: discount factor  
- $\mathcal{P}^a_{ss'}$: 현재 state $s$ 에서 다음 state $s'$ 로의 전이 확률(action $a$ 를 수행했을 때)  
- $v_k(s')$: 다음 state $s'$ 에서의 value fn.

### Example: Small Gridworld  

<div align="center">
<img src="../assets/img/250907_rl3/smallgridworld.png" width="400">
</div>

**[ 조건 ]**  
- undiscounted episodic MDP ($\gamma = 1$)  
- 1, ..., 14는 nonterminal state들
- 회색 박스들이 terminal state들
- 그리드 밖을 나가는 action을 수행할 경우, 원래의 위치로 되돌아오게 됨
- terminal state에 도달하기 전까지 모든 transition의 reward는 -1
- agent는 아래의 랜덤 policy를 따름 (랜덤 policy에서부터 시작하더라도, 결국 최적의 policy에 도달)

$$ \pi(n|\cdot) = \pi(e|\cdot) = \pi(s|\cdot) = \pi(w|\cdot) = 0.25 $$

<div align="center">
  <img src="../assets/img/250907_rl3/smallgridworld_1.png" width="300">
  <img src="../assets/img/250907_rl3/smallgridworld_2.png" width="300">
</div>

계산 예시: north, east, south, west 순서대로 계산  
`k=2, box 1` 0.25(-1.0-1.0) + 0.25(-1.0-1.0) + 0.25(-1.0-1.0) + 0.25(-1.0+0.0) = -1.75 $\approx$ **-1.7**  
`k=2, box 5` 0.25(-1.0-1.0) + 0.25(-1.0-1.0) + 0.25(-1.0-1.0) + 0.25(-1.0-1.0) = **-2.0**  
`k=3, box 1` 0.25(-1.0-1.7) + 0.25(-1.0-2.0) + 0.25(-1.0-2.0) + 0.25(-1.0+0.0) = -2.425 $\approx$ **-2.4**  
(그림을 보면, value fn.이 전부 업데이트가 되지 않았음에도 policy는 최적을 찾은 상태인데, 이는 뒤에 더 자세히 설명될 것~)

## Policy Iteration
### How to Improve Policy  
policy $\pi$ 가 주어졌을 때, 해당 policy의 value fn. $v_\pi$ 를 계산  

$$v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + ... | S_t=s ]$$

계산해서 얻은 $v_\pi$ 에 따라서 매 스텝마다 greedy하게 policy를 업데이트

$$\pi ' = \text{greedy}(v_\pi)$$

Small Gridworld에서 봤던 예제에서도 $\pi ' = \pi^*$ 로 수렴함을 확인  
실제 문제들에 접목했을 때는 더 많은 iteration을 거쳐야 하지만, 어떠한 경우에도 policy iteration은 반드시 최적의 policy $\pi^*$ 로 수렴함

<div align="center">
  <img src="../assets/img/250907_rl3/policy_iter.png" width="400">
</div>

policy evaluation과 policy improvement를 반복적으로 수행  
- policy evaluation: $v_\pi$ 를 계산
- policy improvement: 계산한 $v_\pi$ 를 통해 greedy하게 policy를 업데이트

### Policy Improvement
**[ 반드시 최적의 value fn. & policy로 수렴하는 것을 증명 ]**  
- deterministic policy 가정: $a = \pi(s)$  
- 매 스텝마다 greedy하게 policy를 업데이트: $\pi'(s) = \operatorname*{argmax}\limits_{a \in \mathcal{A}} q_\pi(s,a)$  

ㅎ.. TBU


## Value Iteration


## Extensions to Dynamic Programming


## Contraction Mapping