from torchsummary import summary
from models.swin_unetr import SwinUNETR
from models.simmim import SwinSimMIM
from models.swin_unetr_multi_scale_decoder_fusion import SwinUNETR_Multi_Decoder_Fusion




def main():
    Swin_SimMIM = SwinSimMIM(
        img_size=(128, 128, 128),
        in_channels=1,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
        pretrained=None,
        revise_keys=[],
        masking_ratio=0.75)
    Swin_UNETR = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=1,
        out_channels=1,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        feature_size=48,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
        pretrained=None,
        revise_keys=[]
    )
    Swin_UNETR_MultiDecode_Fusion = SwinUNETR_Multi_Decoder_Fusion(
        img_size=(128, 128, 128),
        in_channels=1,
        out_channels=1,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        feature_size=48,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
        pretrained=None,
        revise_keys=[],
        num_decoder=3,
        overall_pretrained=None
    )
    print('Swin_SimMIM')
    summary(Swin_SimMIM.cuda(), (1, 128, 128, 128), batch_size=1, device="cuda")
    print('Swin_UNETR')
    summary(Swin_UNETR.cuda(), (1, 128, 128, 128), batch_size=1, device="cuda")
    print('Swin_UNETR_MultiDecode_Fusion')
    summary(Swin_UNETR_MultiDecode_Fusion.cuda(), (1, 128, 128, 128), batch_size=1, device="cuda")


if __name__ == '__main__':
    main()