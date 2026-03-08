


# LPTN4k  加载原模型的LPTN参数
python realesrgan/infer_skin.py \
    --image_path /root/autodl-fs/skin/facealign_01src \
    --trainopt /root/autodl-fs/vcolor/RealFaceAI/options/skinpair_00.yml \
    --model_path /root/autodl-fs/vcolor/Real-ESRGAN/experiments/train_RealESRNetx2plus_1000k_B12G4/models/net_g_15000.pth


python realesrgan/infer_skin.py \
    --image_path /autodl-fs/data/skin/testdata \
    --trainopt /root/autodl-fs/vcolor/RealFaceAI/options/skinpair_00.yml \
    --model_path /root/autodl-fs/vcolor/RealFaceAI/experiments/face_hd_03_freqloss_gan/models/net_g_10000.pth

 