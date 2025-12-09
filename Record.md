## BSQ reproduction
* imagenet_256x256_bsqvit_b16g16_stylegan_f8_fp16
* 0.25 bpp
    * imagenet
    PSNR: 26.5282 (±3.6989)
    SSIM: 0.7931 (±0.1007)
    MS-SSIM: 0.9425 (±0.0311)
    LPIPS (AlexNet):    0.0835 (±0.0343)
    FID: 1.6496

    * coco
    PSNR: 26.1550 (±3.3186)
    SSIM: 0.7986 (±0.0858)
    MS-SSIM: 0.9454 (±0.0252)
    LPIPS (AlexNet): 0.0829 (±0.0279)
    FID: 7.0344

* imagenet_256x256_bsqvit_b32g32_stylegan_f8_fp16
* 0.5 bpp
    * imagenet
    PSNR: 28.4454 (±3.7882)
    SSIM: 0.8522 (±0.0761)
    MS-SSIM: 0.9640 (±0.0214)
    LPIPS (AlexNet): 0.0508 (±0.0243)
    FID: 0.7009

    * coco
    PSNR: 28.1940 (±3.3772)
    SSIM: 0.8584 (±0.0649)
    MS-SSIM: 0.9666 (±0.0169)
    LPIPS (AlexNet): 0.0498 (±0.0196)
    FID: 4.5875

* imagenet_256x256_bsqvit_b64g64_stylegan_f8_fp16
* 1.0 bpp
    * imagenet
    PSNR: 31.6082 (±3.7949)
    SSIM: 0.9149 (±0.0446)
    MS-SSIM: 0.9830 (±0.0098)
    LPIPS (AlexNet): 0.0270 (±0.0137)
    FID: 0.3794

    * coco
    PSNR: 31.3809 (±3.3782)
    SSIM: 0.9187 (±0.0378)
    MS-SSIM: 0.9842 (±0.0079)
    LPIPS (AlexNet): 0.0262 (±0.0110)
    FID: 2.8354

## Gaussian VAE
* 1e-7

* 1.1756
kl mean: 18.8118 (±0.2375)
kl min: 4.5879 (±1.0414)
kl max: 44.8097 (±3.0475)
PSNR: 32.0635 (±4.1938)
SSIM: 0.9062 (±0.0558)
MS-SSIM: 0.9860 (±0.0096)
LPIPS (AlexNet): 0.0243 (±0.0145)
FID: 0.3749

* 3e-7

* 0.773 bpp
kl mean: 12.3605 (±0.2905)
kl min: 0.0175 (±0.0306)
kl max: 31.8528 (±1.8715)
PSNR: 31.4198 (±4.0925)
SSIM: 0.8966 (±0.0589)
MS-SSIM: 0.9832 (±0.0109)
LPIPS (AlexNet): 0.0299 (±0.0187)
FID: 0.6369

* 1e-6

* 0.417 bpp
kl mean: 6.6748 (±0.3027)
kl min: 0.0088 (±0.0092)
kl max: 26.4736 (±1.8136)
PSNR: 30.8869 (±3.4762)
SSIM: 0.8879 (±0.0552)
MS-SSIM: 0.9795 (±0.0113)
LPIPS (AlexNet): 0.0331 (±0.0151)
FID: 0.5757

* 3e-6

* 0.195 bpp
kl mean: 3.1188 (±0.1706)
kl min: 0.0004 (±0.0002)
kl max: 29.7087 (±4.9009)
PSNR: 28.5935 (±3.3242)
SSIM: 0.8297 (±0.0789)
MS-SSIM: 0.9604 (±0.0207)
LPIPS (AlexNet): 0.0561 (±0.0228)
FID: 0.8789

* 7.0e-7

* bpp: 0.496

kl mean: 7.9483 (±0.3255)
kl min: 0.0025 (±0.0015)
kl max: 28.2110 (±1.7945)
PSNR: 31.6498 (±3.8361)
SSIM: 0.9001 (±0.0538)
MS-SSIM: 0.9824 (±0.0104)
LPIPS (AlexNet): 0.0304 (±0.0164)
FID: 0.8673

* 2.0e-7

* bpp: 0.885

kl mean: 14.1607 (±0.3575)
kl min: 0.0110 (±0.0213)
kl max: 36.2185 (±3.0309)
PSNR: 32.3504 (±4.1985)
SSIM: 0.9104 (±0.0525)
MS-SSIM: 0.9862 (±0.0093)
LPIPS (AlexNet): 0.0236 (±0.0146)
FID: 0.5204


* 18e-7 30e-7 23e-7 
* 0.296 0.195 

* imagenet_256x256_gaussianvit_kl2.2e-6_target_stylegan_f8_fp16
* kl
    * imagenet
    kl mean: 15.8532 (±0.0908)
    kl min: 7.7180 (±1.4164)
    kl max: 16.6010 (±0.0207)
    PSNR: 28.4442 (±3.9527)
    SSIM: 0.8374 (±0.0846)
    MS-SSIM: 0.9633 (±0.0219)
    LPIPS (AlexNet): 0.0545 (±0.0262)
    FID: 0.7931
    * coco


* mle (18/16)
    PSNR: 27.7093 (±3.8011)
    SSIM: 0.8097 (±0.0942)
    MS-SSIM: 0.9593 (±0.0236)
    LPIPS (AlexNet): 0.0647 (±0.0331)
    FID: 1.7287

* mle (16/16)
    * imagenet

    PSNR: 27.6032 (±3.9820)
    SSIM: 0.8023 (±0.0992)
    MS-SSIM: 0.9565 (±0.0260)
    LPIPS (AlexNet): 0.0719 (±0.0407)
    FID: 2.1205

    * coco
    PSNR: 27.1061 (±3.5575)
    SSIM: 0.8039 (±0.0856)
    MS-SSIM: 0.9581 (±0.0213)
    LPIPS (AlexNet): 0.0732 (±0.0329)
    FID: 9.2862


* imagenet_256x256_gaussianvit_kl7e-7_target_stylegan_f8_fp16_lr_2e-7
* kl
    * imagenet

    kl mean: 15.8654 (±0.1002)
    kl min: 6.6559 (±1.8228)
    kl max: 16.6796 (±0.0980)
    PSNR: 31.1898 (±4.1149)
    SSIM: 0.8965 (±0.0595)
    MS-SSIM: 0.9814 (±0.0123)
    LPIPS (AlexNet): 0.0330 (±0.0189)
    FID: 0.7575

    * coco
    PSNR: 30.9251 (±3.6194)
    SSIM: 0.9013 (±0.0509)
    MS-SSIM: 0.9828 (±0.0099)
    LPIPS (AlexNet): 0.0307 (±0.0148)
    FID: 3.5899

* bpp 0.5 mle (16/8) 
    * imagenet
    PSNR: 30.4276 (±4.0176)
    SSIM: 0.8817 (±0.0662)
    MS-SSIM: 0.9783 (±0.0140)
    LPIPS (AlexNet): 0.0369 (±0.0202)
    FID: 0.5918

    * coco
    PSNR: 30.1833 (±3.5775)
    SSIM: 0.8877 (±0.0563)
    MS-SSIM: 0.9799 (±0.0113)
    LPIPS (AlexNet): 0.0347 (±0.0157)
    FID: 3.6166

* imagenet_256x256_gaussianvit_kl3e-6_target_stylegan_f8_fp16
* kl
* imagenet
    PSNR: 27.3291 (±3.7674)
    SSIM: 0.8068 (±0.0950)
    MS-SSIM: 0.9525 (±0.0267)
    LPIPS (AlexNet): 0.0656 (±0.0301)
    FID: 1.1033
* mle (16/16)

    PSNR: 26.9890 (±3.7961)
    SSIM: 0.7782 (±0.1054)
    MS-SSIM: 0.9514 (±0.0277)
    LPIPS (AlexNet): 0.0802 (±0.0424)
    FID: 2.9026

* imagenet_256x256_gaussianvit_kl1.5e-7_target_stylegan_f8_fp16_lr_2e-7
* kl
    * imagenet
    kl mean: 15.9995 (±0.0368)
    kl min: 12.8340 (±0.7421)
    kl max: 17.2370 (±0.4110)
    PSNR: 32.0587 (±4.1520)
    SSIM: 0.9094 (±0.0524)
    MS-SSIM: 0.9860 (±0.0093)
    LPIPS (AlexNet): 0.0231 (±0.0140)
    FID: 0.3978

    * coco

    PSNR: 31.8358 (±3.6263)
    SSIM: 0.9141 (±0.0445)
    MS-SSIM: 0.9871 (±0.0075)
    LPIPS (AlexNet): 0.0216 (±0.0111)
    FID: 2.7211

* bpp 1.0 mle (16/4)
    * imagenet
    PSNR: 31.7135 (±4.0517)
    SSIM: 0.9032 (±0.0542)
    MS-SSIM: 0.9849 (±0.0097)
    LPIPS (AlexNet): 0.0239 (±0.0143)
    FID: 0.3489

    * coco
    PSNR: 31.5053 (±3.5768)
    SSIM: 0.9089 (±0.0456)
    MS-SSIM: 0.9860 (±0.0079)
    LPIPS (AlexNet): 0.0226 (±0.0113)
    FID: 2.7030

* imagenet_256x256_gaussianvit_kl2.0e-6_target_stylegan_f8_fp16_c8
  * bpp 0.25 mle (16/8)
    * imagenet
    * mle
        PSNR: 27.2994 (±3.8096)
        SSIM: 0.7951 (±0.0991)
        MS-SSIM: 0.9536 (±0.0266)
        LPIPS (AlexNet): 0.0680 (±0.0337)
        FID: 1.9529

    * det
        PSNR: 27.8015 (±3.9383)
        SSIM: 0.8164 (±0.0948)
        MS-SSIM: 0.9575 (±0.0253)
        LPIPS (AlexNet): 0.0634 (±0.0292)
        FID: 1.0730

    * coco
        * mle
        PSNR: 27.1272 (±3.4516)
        SSIM: 0.8085 (±0.0827)
        MS-SSIM: 0.9573 (±0.0213)
        LPIPS (AlexNet): 0.0658 (±0.0264)
        FID: 7.9228

        * det
        PSNR: 27.4835 (±3.5249)
        SSIM: 0.8240 (±0.0806)
        MS-SSIM: 0.9601 (±0.0205)
        LPIPS (AlexNet): 0.0622 (±0.0237)
        FID: 5.6737

# Lambert 

* imagenet_256x256_lambert_t_16_g_16_stylegan_f8_fp16
    * imagenet
        * mean
        PSNR: 28.7912 (±3.8558)
        SSIM: 0.8366 (±0.0891)
        MS-SSIM: 0.9699 (±0.0193)
        LPIPS (AlexNet): 0.0532 (±0.0243)
        FID: 1.4146

        * max
        PSNR: 25.0870 (±2.8869)
        SSIM: 0.7740 (±0.0993)
        MS-SSIM: 0.9319 (±0.0332)
        LPIPS (AlexNet): 0.0974 (±0.0459)
        FID: 3.0253

        * kl
        PSNR: 27.6968 (±3.8811)
        SSIM: 0.8179 (±0.0928)
        MS-SSIM: 0.9579 (±0.0244)
        LPIPS (AlexNet): 0.0607 (±0.0273)
        FID: 1.0154

        * det 1.0
        PSNR: 27.1838 (±3.8090)
        SSIM: 0.7985 (±0.0982)
        MS-SSIM: 0.9525 (±0.0267)
        LPIPS (AlexNet): 0.0654 (±0.0292)
        FID: 1.2480

        * mrc
        PSNR: 26.9403 (±3.7870)
        SSIM: 0.7905 (±0.1002)
        MS-SSIM: 0.9496 (±0.0279)
        LPIPS (AlexNet): 0.0691 (±0.0306)
        FID: 1.5216
    
        * orc
        PSNR: 27.2102 (±3.6493)
        SSIM: 0.7984 (±0.0952)
        MS-SSIM: 0.9518 (±0.0269)
        LPIPS (AlexNet): 0.0694 (±0.0304)
        FID: 1.3799

        * mle
        PSNR: 26.6400 (±3.7173)
        SSIM: 0.7736 (±0.1060)
        MS-SSIM: 0.9475 (±0.0287)
        LPIPS (AlexNet): 0.0829 (±0.0387)
        FID: 3.2291

    * coco
        * kl
        PSNR: 27.3936 (±3.4710)
        SSIM: 0.8245 (±0.0790)
        MS-SSIM: 0.9604 (±0.0197)
        LPIPS (AlexNet): 0.0599 (±0.0224)
        FID: 5.4821

        * det
        PSNR: 26.8995 (±3.4440)
        SSIM: 0.8068 (±0.0836)
        MS-SSIM: 0.9553 (±0.0217)
        LPIPS (AlexNet): 0.0650 (±0.0243)
        FID: 6.4683

* imagenet_256x256_lambert_t_16_g_8_stylegan_f8_fp16
    * imagenet
        * det
        PSNR: 29.8843 (±3.9210)
        SSIM: 0.8726 (±0.0702)
        MS-SSIM: 0.9755 (±0.0152)
        LPIPS (AlexNet): 0.0380 (±0.0194)
        FID: 0.5576

        * mle
        PSNR: 29.7025 (±3.8571)
        SSIM: 0.8680 (±0.0729)
        MS-SSIM: 0.9751 (±0.0154)
        LPIPS (AlexNet): 0.0396 (±0.0200)
        FID: 0.6488

    * coco

* imagenet_256x256_lambert_t_16_g_4_stylegan_f8_fp16
    * imagenet
        * det

* imagenet_256x256_ta_t_16_g_16_stylegan_f8_fp16_lr_2e-7
* det 16/16 

imagenet

PSNR: 27.8856 (±3.8782)
SSIM: 0.8233 (±0.0898)
MS-SSIM: 0.9594 (±0.0238)
LPIPS (AlexNet): 0.0607 (±0.0289)
FID: 0.9325

* coco

PSNR: 27.5577 (±3.4682)
SSIM: 0.8296 (±0.0772)
MS-SSIM: 0.9617 (±0.0194)
LPIPS (AlexNet): 0.0600 (±0.0239)
FID: 5.3053
