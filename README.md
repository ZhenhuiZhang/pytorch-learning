# pytorch-learning

## pytorch1.4
[Document](https://pytorch.org/)

1. A、B、C、D、E五人在某天夜里合伙捕鱼 最后疲惫不堪各自睡觉
2. 第二天A第一个醒来 他将鱼分为5份 扔掉多余的1条 拿走自己的一份
3. B第二个醒来 也将鱼分为5份 扔掉多余的1条 拿走自己的一份
4. 然后C、D、E依次醒来也按同样的方式分鱼 问他们至少捕了多少条鱼

fish = 6
attemp = 1
while True:
    total = fish
    enough = True
    t = []
    for _ in range(5):
        t.append(total)
        if (total - 1) % 5 == 0:
            total = (total - 1) // 5 * 4
        else:
            enough = False
            break
    if enough:
        if attemp != 2:
            enough = False
            attemp+=1
            print(fish)
        break  
    fish += 5