import csv
from tkinter import simpledialog
import pygame
import random
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt  # 导入matplotlib库

# 迷宫尺寸和复杂度（可以调整）
maze_height = 12
maze_width = 12
# 初始化迷宫（0为通道，1为障碍物）
maze = np.zeros((maze_height, maze_width))
# 随机生成一些障碍物，1表示障碍物
for i in range(maze_height):
    for j in range(maze_width):
        if random.random() < 0.3:  # 30%的概率是障碍物
            maze[i][j] = 1

# 迷宫生成算法
def generate_maze(height, width):
    # 定义迷宫中的墙壁（按格子生成）
    walls = set((x, y) for x in range(width) for y in range(height) if (x % 2 == 0 or y % 2 == 0))

    # 随机选择一个起始点（必须是一个通道）
    start_index = random.randint(0, len(walls) - 1)
    current_cell = list(walls)[start_index]  # 随机选择一个起始点
    visited = set()
    stack = [current_cell]
    walls.remove(current_cell)  # 从墙壁集合中移除起始点

    while stack:
        current_cell = stack[-1]
        visited.add(current_cell)
        neighbors = []

        # 这里的循环处理上下左右4个方向
        for i in range(4):
            dx, dy = 1, 0
            if i == 1:
                dx, dy = 0, 1
            elif i == 2:
                dx, dy = -1, 0
            elif i == 3:
                dx, dy = 0, -1

            next_cell = (current_cell[0] + dx * 2, current_cell[1] + dy * 2)

            # 确保不会越界并且未访问过且是墙壁
            if 0 <= next_cell[0] < height and 0 <= next_cell[1] < width:
                if next_cell not in visited and next_cell in walls:
                    neighbors.append(next_cell)

        # 如果存在邻居，继续生成
        if neighbors:
            next_index = random.randint(0, len(neighbors) - 1)
            next_cell = neighbors[next_index]
            stack.append(next_cell)
            # 移除当前单元格和下一个单元格之间的墙壁
            wall = ((current_cell[0] + next_cell[0]) // 2, (current_cell[1] + next_cell[1]) // 2)
            walls.discard(wall)  # 使用discard来避免找不到问题
            maze[wall[0], wall[1]] = 0  # 设置中间的墙壁为通道
        else:
            stack.pop()

# 生成迷宫并确保起点和终点之间有路径
generate_maze(maze_height, maze_width)

# 设置起始位置和目标位置
start_position = (0, 0)
goal_position = (maze_height - 1, maze_width - 1)

# 添加炸弹位置
num_bombs = 7  # 设置炸弹数量
bombs = set()
while len(bombs) < num_bombs:
    bomb = (random.randint(0, maze_height - 1), random.randint(0, maze_width - 1))
    if maze[bomb[0], bomb[1]] == 0 and bomb != start_position and bomb != goal_position:
        bombs.add(bomb)

# Q学习相关参数
learning_rate = 0.1
discount_factor = 0.8
epsilon = 0.9
epsilon_min = 0.01
epsilon_decay = 0.99
num_episodes = 200

# 定义动作空间：上、下、左、右
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 初始化Q表
Q = np.zeros((maze_height, maze_width, len(actions)))

# 记录每轮训练的最优路径
paths = []
reward_history = []

# 定义奖励函数
def reward(position):
    if position == goal_position:
        return 100  # 到达目标时的奖励
    elif maze[position] == 1:  # 遇到障碍物
        return -100  # 障碍物的惩罚
    elif position in bombs:  # 遇到炸弹
        return -50  # 炸弹的惩罚
    else:
        return -1  # 普通空白格子的惩罚

# 定义行动选择函数（epsilon-greedy策略）
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))  # 探索
    else:
        return np.argmax(Q[state[0], state[1]])  # 利用Q表中最大的Q值对应的动作

# 更新Q表
def update_Q(state, action, r, next_state):
    best_next_action = np.argmax(Q[next_state[0], next_state[1]])  # 下一个状态的最佳动作
    Q[state[0], state[1], action] += learning_rate * (
            r + discount_factor * Q[next_state[0], next_state[1], best_next_action] - Q[state[0], state[1], action])

def move(state, action):
    new_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    # 保证不出界
    new_state = (max(0, min(maze_height - 1, new_state[0])), max(0, min(maze_width - 1, new_state[1])))

    # 如果目标位置是障碍物或尝试回到起点，则不能移动
    if maze[new_state[0], new_state[1]] == 1 or new_state == start_position:
        return state  # 返回当前状态，表示不能移动

    return new_state

# 训练过程
best_path_length = float('inf')
best_path_episode = None
for episode in range(num_episodes):
    state = start_position
    done = False
    total_reward = 0
    path = [state]  # 跟踪每一轮的路径

    while not done:
        action = choose_action(state)
        next_state = move(state, action)  # 执行动作，得到下一个状态
        r = reward(next_state)  # 获得奖励

        # 更新Q表
        update_Q(state, action, r, next_state)

        state = next_state  # 更新当前状态
        path.append(state)

        total_reward += r

        # 如果达到目标，则训练结束
        if state == goal_position:
            done = True

    paths.append(path)  # 记录每轮的路径
    reward_history.append(total_reward)  # 记录每轮的总奖励

    # 每一轮后，降低epsilon，减少探索
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    # 找到最优路径
    if len(path) < best_path_length:
        best_path_length = len(path)
        best_path_episode = episode

# 打印最优路径的轮数
if best_path_episode is not None:
    print(f"Best path found in episode {best_path_episode + 1} with length {best_path_length}")

# 打印最优路径对应的Q表
print("\nQ-table for best path:")
# 获取最优路径的坐标列表
best_path = paths[best_path_episode]
for position in best_path:
    i, j = position
    if maze[i][j] == 0:  # 只打印通道的Q值
        print(f"Position ({i},{j}):", end=" ")
        for a in range(len(actions)):
            print(f"Action {a}: {Q[i, j, a]:.2f}", end="  ")
        print()  # 换行
    else:
        print(f"Position ({i},{j}): Wall")  # 墙壁位置不打印Q值

# 可视化奖励折线图
plt.figure(figsize=(10, 6))
plt.plot(reward_history, label="Total Reward per Episode", color='b')
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-learning Reward Progression")
plt.legend()
plt.grid(True)
plt.show()

# tkinter配置
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 用pygame展示迷宫
pygame.init()

# 加载炸弹和机器人图片
try:
    bomb_image = pygame.image.load('bomb.png')
    bomb_image = pygame.transform.scale(bomb_image, (40, 40))  # 调整炸弹大小
except pygame.error as e:
    print("Failed to load bomb image:", e)
    bomb_image = None

try:
    robot_image = pygame.image.load('robot.png')
    robot_image = pygame.transform.scale(robot_image, (40, 40))  # 调整机器人的大小与迷宫格子匹配
except pygame.error as e:
    print("Failed to load robot image:", e)
    robot_image = None

# 设置单元格大小
cell_size = 40
screen = pygame.display.set_mode((maze_width * cell_size, maze_height * cell_size))
pygame.display.set_caption("Maze")

# 绘制迷宫
def draw_maze():
    screen.fill((255, 255, 255))  # 背景色为白色
    for row in range(maze_height):
        for col in range(maze_width):
            color = (255, 255, 255) if maze[row, col] == 0 else (0, 0, 0)
            if maze[row, col] == 0:  # 通道绘制为圆形
                pygame.draw.circle(screen, color,
                                   (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2),
                                   cell_size // 2)
            else:  # 障碍物用矩形绘制
                pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))

    # 绘制炸弹
    if bomb_image:
        for bomb in bombs:
            bomb_x, bomb_y = bomb
            screen.blit(bomb_image, (bomb_y * cell_size, bomb_x * cell_size))

        # 绘制起点和终点
        pygame.draw.rect(screen, (255, 0, 0),
                         (start_position[1] * cell_size, start_position[0] * cell_size, cell_size, cell_size))
        pygame.draw.rect(screen, (0, 255, 0),
                         (goal_position[1] * cell_size, goal_position[0] * cell_size, cell_size, cell_size))
    pygame.display.flip()

# 动态显示机器人移动
def display_path(path_to_show):
    ball_pos = start_position
    clock = pygame.time.Clock()
    running = True
    index = 0

    while running and index < len(path_to_show):
        draw_maze()

        if robot_image:
            ball_x, ball_y = path_to_show[index]
            screen.blit(robot_image, (ball_y * cell_size, ball_x * cell_size))

        pygame.display.flip()

        # 检查是否碰到炸弹
        if (ball_x, ball_y) in bombs:
            pygame.time.delay(500)
            print("Game Over! You hit a bomb.")
            pygame.time.delay(1000)
            running = False

        pygame.time.delay(100)  # 减少延迟，增加动画流畅度
        # 继续显示机器人的动画
        index += 1

        # 检查是否到达目标
        if (ball_x, ball_y) == goal_position:
            print("Congratulations! You reached the goal!")
            pygame.time.delay(500)  # 停顿一会
            running = False

        clock.tick(10)  # 控制更新帧率，调整为适当的速度，值越小越快

# 主循环：允许用户选择查看不同的轮次路径
running = True
while running:
    # 选择展示的轮数
    episode_to_show = simpledialog.askinteger("选择展示的轮数", f"请输入训练轮数（1-{num_episodes}）：", minvalue=1, maxvalue=num_episodes)

    if episode_to_show:
        path_to_show = paths[episode_to_show - 1]
        print(f"Path for episode {episode_to_show}:")
        for step in path_to_show:
            print(step)

        display_path(path_to_show)

    # 检查是否退出
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 询问用户是否继续查看另一个轮次的路径
    continue_prompt = simpledialog.askstring("继续查看?", "是否继续查看另一个轮次的路径？(yes/no)")
    if continue_prompt.lower() != 'yes':
        running = False

pygame.quit()  # 退出pygame
