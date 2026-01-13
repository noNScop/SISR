from engine import *

def load_model(model_name: str, scaling_factor: int, device: str):
    if model_name == "Bicubic":
        return BicubicInterpolation(scaling_factor)
    
    elif model_name == "FSRCNN" and os.path.isfile(f"../model_checkpoints/FSRCNN/X{scaling_factor}.pth"):
        model = FSRCNN(scaling_factor).to(device)
        checkpoint = torch.load(f'../model_checkpoints/FSRCNN/X{scaling_factor}.pth', map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    elif model_name == "SRResNet" and os.path.isfile(f"../model_checkpoints/SRResNet/X{scaling_factor}.pth"):
        model = SRResNet(scaling_factor).to(device)
        checkpoint = torch.load(f'../model_checkpoints/SRResNet/X{scaling_factor}.pth', map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    elif model_name == "SRGAN" and os.path.isfile(f"../model_checkpoints/SRGAN/X{scaling_factor}.pth"):
        model = SRGAN(scaling_factor).to(device)
        checkpoint = torch.load(f'../model_checkpoints/SRGAN/X{scaling_factor}.pth', map_location=torch.device(device))
        model.load_state_dict(checkpoint['generator_state_dict'])
        return model
    
    elif model_name == "RCAN" and os.path.isfile(f"../model_checkpoints/RCAN/X{scaling_factor}.pth"):
        model = RCAN(scaling_factor).to(device)
        checkpoint = torch.load(f'../model_checkpoints/RCAN/X{scaling_factor}.pth', map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    else:
        print(f"X{scaling_factor} is unavaiable for {model_name}")
        return None
    
def choose_scale(model_name: str, device: str):
    while True:
        print(f"\n===== Choose upscaling Factor For {model_name} =====")
        print("1. 2X")
        print("2. 4X")
        print("3. 8X")
        print("4. Change the model")

        choice = input("Select an option (1-4): ")

        if choice == "1":
            return load_model(model_name, 2, device)
        elif choice == "2":
            return load_model(model_name, 4, device)
        elif choice == "3":
            return load_model(model_name, 8, device)
        elif choice == "4":
            return None
        else:
            print("Invalid option. Please choose 1-4.")
    
def choose_model(device: str):
    while True:
        print("\n===== Choose Super Resolution Architecture =====")
        print("1. Bicubic Interpolation")
        print("2. FSRCNN")
        print("3. SRResNet")
        print("4. SRGAN")
        print("5. RCAN")

        choice = input("Select an option (1-4): ")
        model = None

        if choice == "1":
            model = choose_scale("Bicubic", device)

        elif choice == "2":
            model = choose_scale("FSRCNN", device)

        elif choice == "3":
            model = choose_scale("SRResNet", device)

        elif choice == "4":
            model = choose_scale("SRGAN", device)
        
        elif choice == "5":
            model = choose_scale("RCAN", device)

        else:
            print("Invalid option. Please choose 1-4.")

        if model is not None:
            model.eval()
            return model
        
def is_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def main():
    model = choose_model(device)

    while True:
        print("\n===== Super Resolution Terminal App =====")
        print(f"1. Change model [{model.name}]")
        print(f"2. Change scale [{model.scale}]")
        print("3. Input image")
        print("4. Exit")

        choice = input("Select an option (1-4): ")

        if choice == "1":
            model = choose_model(device)

        elif choice == "2":
            new_model = choose_scale(model.name, device)
            if new_model != None:
                model = new_model

        elif choice == "3":
            image_path = input("Enter path to input image: ")
            if is_image(image_path):
                model(image_path)
                print(f"Scaled image succesfully stored in {os.getcwd()}/output")
            else:
                print("Image doesn't exist")

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid option. Please choose 1-4.")

if __name__ == "__main__":
    main()