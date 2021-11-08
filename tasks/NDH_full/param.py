import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_type', type=str, required=True,
                    help='planner_path, player_path, or trusted_path')
parser.add_argument('--feedback', type=str, required=True,
                    help='teacher or sample')
parser.add_argument('--eval_type', type=str, required=True,
                    help='val or test')
parser.add_argument('--blind', action='store_true', required=False,
                    help='whether to replace the ResNet encodings with zero vectors at inference time')
parser.add_argument('--maxInput', type=int, default=512, help="max input instruction")
parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)
parser.add_argument('--angle_feat_size', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_view', type=int, default=36)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--featdropout', type=float, default=0.3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--ignoreid', type=int, default=-100)
parser.add_argument('--prefix', type=str, default="v1", required=True)
parser.add_argument('--vlnbert', type=str, default="prevalent")
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--nr', type=int, default=0)
parser.add_argument('--server', type=str, default="server")
parser.add_argument('--feat_type', type=str, default="resnet_clip")
parser.add_argument('--reinforce', action='store_true', required=False)
args = parser.parse_args()
