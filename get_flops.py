from timesformer.utils.parser import load_config, parse_args
import timesformer.utils.misc as misc
from timesformer.models import build_model
import logging
logging.basicConfig(level=logging.INFO)


def main():
    args = parse_args()
    cfg = load_config(args)
    model = build_model(cfg)
    misc.log_model_info(model, cfg, use_train_input=True)


if __name__ == "__main__":
    main()
