import time
import os
import pyautogui
from PIL import ImageGrab


import time
import os
from PIL import ImageGrab

def crop_screen_and_save_delay(bbox, save_path, delay=3):
    """
    bbox: [x1, y1, x2, y2]
    save_path: 保存路径（含文件名）
    delay: 延时秒数
    """
    print(f"{delay} 秒后开始截屏...")
    time.sleep(delay)

    x1, y1, x2, y2 = bbox
    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

    print(f"已保存：{save_path}")


def click_hold_and_capture(
    click_pos,     # [a1, b1]
    bbox,          # [x1, y1, x2, y2]
    save_path,
    delay=3,
    button="right" # "left" / "right"
):
    """
    1. 在 click_pos 处按下鼠标（不松开）
    2. 延时 delay 秒
    3. 截取 bbox
    4. 松开鼠标
    """

    a1, b1 = click_pos
    x1, y1, x2, y2 = bbox

    print(f"鼠标按下({button}) @ {click_pos}")
    pyautogui.mouseDown(x=a1, y=b1, button=button)

    print(f"等待 {delay} 秒后截屏...")
    time.sleep(delay)

    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"已保存：{save_path}")

    print("松开鼠标")
    pyautogui.mouseUp(x=a1, y=b1, button=button)


if  __name__=='__main__':

    # click_pos = [800, 500]              # 鼠标按下位置
    # bbox = [700, 400, 1100, 800]        # 截屏区域


    click_pos = [998, 2360]              # 鼠标按下位置
    bbox = [0, 600, 1030, 1836]        # 截屏区域

    crop_screen_and_save_delay(bbox, 
                               "./output/gt_sy2.png",
                                 delay=3)

    click_hold_and_capture(
        click_pos=click_pos,
        bbox=bbox,
        save_path="./output/src_sy2.png",
        delay=3,
        button="left"
    )


