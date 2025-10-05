import pygame, torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

pygame.init() 
SCRSIZE = (750, 500)
FPS = 60
DOT_SIZE = 15
ERASE_SIZE = 20
DELAY = 1*FPS
BUTTON_RECT = (300, 425, 150, 50)
DRAW_SURFACE = (175, 0, 400, 400)
FONT = pygame.font.Font("freesansbold.ttf", 32)
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

class ConvNet(nn.Module):
   def __init__(self):
       super(ConvNet, self).__init__()
       self.layer1 = nn.Sequential(
           nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # greyscale
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2)
       )
       self.layer2 = nn.Sequential(
           nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2)
       )
       self.fc = nn.Linear(7*7*32, 10)  # 10 Classes (0-9)
       self.m = nn.Softmax(dim=1)
       
   def forward(self, x):
       out = self.layer1(x)
       out = self.layer2(out)
       out = out.reshape(out.size(0), -1)  # Flatten
       out = self.fc(out)
       out = self.m(out)
       return out
            

class Window:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
        transforms.Resize((28, 28)),                 # Resize to MNIST dimensions
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))   # Normalize with MNIST dataset values
    ])

    MODEL = ConvNet()
    MODEL.load_state_dict(torch.load("mnist_cnn.pth"))

    BUTTON_LABEL = FONT.render("CHECK", True, (255, 255, 255))
    
    def __init__(self):
        self.screen = pygame.display.set_mode(SCRSIZE)
        self.running = True
        self.draw_state = 0       # 0: None, 1: Draw, 2: Erase, 3: Clear
        self.clock = pygame.time.Clock()
        self.dots =  []
        self.result  = ""
        self.prob = 0
    
    def main(self):
        prev = 0
        timer = DELAY
        pygame.display.set_caption("NumAI")
        
        while self.running:
            
            mousepos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.draw_state = 1
                    elif event.button == 2:
                        self.dots = []
                    elif event.button == 3:
                        self.draw_state = 2
                    
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.draw_state = 0
            
            # Add Dots
            if self.draw_state == 1 and mousepos[0] > DRAW_SURFACE[0] + DOT_SIZE and mousepos[0] < DRAW_SURFACE[0] + DRAW_SURFACE[2] - DOT_SIZE and mousepos[1] < DRAW_SURFACE[3] - DOT_SIZE:
                self.dots.append(mousepos)
            # Erase Dots
            elif self.draw_state == 2:
                for dot in self.dots:
                    if np.sqrt((mousepos[0] - dot[0])**2 + (mousepos[1] - dot[1])**2) < DOT_SIZE + ERASE_SIZE:
                        self.dots.remove(dot)
            
            # Clear Screen
            self.screen.fill((255, 255, 255))
            
            # Draw Draw Surface
            pygame.draw.rect(self.screen, (0, 0, 0), DRAW_SURFACE)

            # Draw Result Label
            if self.result != "":
                label = FONT.render(self.result, True, (0, 0, 0))
                self.screen.blit(label, (SCRSIZE[0]/2 - 10, SCRSIZE[1]-50))
                
                plabel = FONT.render(str(self.prob) + "%", True, (0, 0, 0))
                self.screen.blit(plabel, (SCRSIZE[0]-150, SCRSIZE[1]-50))            
            
            # Draw Dots
            if len(self.dots) > 0:
               for dot in self.dots:
                   pygame.draw.circle(self.screen, (255, 255, 255), dot, DOT_SIZE)

               # Timer to prevent AI from trying to analize every frame
               if timer <= 0:
                  sub = self.screen.subsurface(DRAW_SURFACE)
                  pygame.image.save(sub, "screenshot.jpg")
                  self.evaluate()
                  timer = DELAY
               else:
                  timer -= 1

            else:
               self.result = ""
                
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


    def evaluate(self):
        image = Image.open("screenshot.jpg")
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension: (1, 1, 28, 28)

        # Predict with the model
        with torch.no_grad():
            output = self.MODEL(image)
            predicted = output.argmax(dim=1).item()
            self.result = CLASSES[predicted]
            self.prob = round(output[0][predicted].item()*100, 2)   
      

               
if __name__ == "__main__":
    window = Window()
    window.main()
