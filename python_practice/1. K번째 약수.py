# split ()
# - split(sep='', maxsplit='1')
# - 문자열을 공백 or 어떠한 기준으로 나눌 때 사용
# - 나누어진 값은 리스트에 요소로 저장
# - 리스트로 리턴


a = 'p.y.t.h.o.n 프로그래밍'
print(a.split())
print(a.split('.', 2)) # 문자열을 2번 쪼갠다
print(a.split('.', 6)) # 문자열을 6번 쪼갠다
print(a.split(sep='.', maxsplit=6)) # sep, maxsplit은 생략 가능

import sys
sys.stdin=open('input.txt', 'rt')
n, k = map(int, input().split())
cnt = 0
for i in range(1, n+1):
    if n%i==0:
        # n을 1부터 n까지 나눠서 나머지가 0인 것을 구한다
        cnt+=1
    if cnt==k:
        print(i)
        break
else:
    print(-1)