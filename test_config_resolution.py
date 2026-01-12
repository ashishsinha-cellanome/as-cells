import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def test_config(cfg: DictConfig):
    print("--- Resolved Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    
    print("\n--- Label Map ---")
    if hasattr(cfg.model, 'label_map'):
        print(f"model.label_map: {cfg.model.label_map}")
    else:
        print("model.label_map not found")
        
    print("\n--- Class Remapping ---")
    if hasattr(cfg.data, 'class_remapping'):
        print(f"data.class_remapping: {cfg.data.class_remapping}")
    else:
        print("data.class_remapping not found")

if __name__ == "__main__":
    test_config()
