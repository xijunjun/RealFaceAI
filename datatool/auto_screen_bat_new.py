import time
import os
import pyautogui
from PIL import ImageGrab
from pynput import mouse
import sys

# 全局控制变量
running = True

def on_click(x, y, button, pressed):
    """
    鼠标点击回调函数
    """
    global running
    if pressed and button == mouse.Button.right:
        print("\n[检测到右键点击] 正在停止程序...")
        running = False
        return False  # 返回 False 以停止监听线程

def crop_screen_and_save_delay(bbox, save_path, delay=3):
    if not running: return
    time.sleep(delay)
    x1, y1, x2, y2 = bbox
    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"已保存：{save_path}")

def click_hold_and_capture(click_pos, bbox, save_path, delay=3, button="right"):
    if not running: return
    a1, b1 = click_pos
    x1, y1, x2, y2 = bbox

    print(f"鼠标按下({button}) @ {click_pos}")
    pyautogui.mouseDown(x=a1, y=b1, button=button)
    
    time.sleep(delay)
    
    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    
    pyautogui.mouseUp(x=a1, y=b1, button=button)
    print(f"已保存且松开鼠标：{save_path}")

if __name__ == '__main__':
    # --- 启动鼠标监听器 (非阻塞模式) ---
    listener = mouse.Listener(on_click=on_click)
    listener.start()

    click_pos = [998, 2360]
    bbox = [0, 600, 1030, 1836]
    dstroot = './scshot/gt'
    srcroot = './scshot/src'

    print("程序启动，点击【鼠标右键】可随时退出...")
    time.sleep(2)

    for i in range(0, 200):
        # 检查是否需要退出
        if not running:
            break

        print(f"\n--- 正在处理第 {i} 张 ---")
        pathgt = os.path.join(dstroot, 'skin_' + str(i).zfill(4) + '.png')
        pathlq = os.path.join(srcroot, 'skin_' + str(i).zfill(4) + '.png')

        # 1. 执行第一步截图
        crop_screen_and_save_delay(bbox, pathgt, delay=0.2)

        # 再次检查状态（防止在耗时操作中间卡死）
        if not running: break

        # 2. 执行按住截图
        click_hold_and_capture(
            click_pos=click_pos,
            bbox=bbox,
            save_path=pathlq,
            delay=0.2,
            button="left"
        )
        time.sleep(0.2)

        if not running: break

        # 3. 翻页
        pyautogui.press('right')
        time.sleep(1)

    print("程序已结束。")
    listener.stop() # 确保监听器关闭