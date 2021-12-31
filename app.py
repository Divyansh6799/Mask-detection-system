from kivymd.app import MDApp
from kivy.app import App
from kivymd.uix import screen
from kivymd.uix.label import MDLabel,MDIcon
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDRectangleFlatButton,MDFlatButton
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivymd.uix.dialog import MDDialog
from kivy.core.window import Window
from maskvideo import mask_detect

class Mask_Detection(MDApp):

    def build(self):
        Window.size=(500,500)
        self.theme_cls.primary_palette="Red"
        self.theme_cls.primary_hue="A700"
        self.theme_cls.theme_style="Light"
        screen=Screen()
      
        
        label=MDLabel(text="Face Mask Detection",bold=True,
                      size_hint=(0.5,0.5),
                      font_style='H3',halign='center',
                      pos_hint={'center_x':0.5,'center_y':0.90},
                      font_size="80sp",
                    theme_text_color="Error")
                              
        btn_flat=MDRectangleFlatButton(text="Click Here",
                                       pos_hint={'center_x':0.50,'center_y':0.15},
                                       md_bg_color=self.theme_cls.primary_light,
                                       on_release=self.start_dialog)
       
        
        img=Image(source='background.jpg',opacity=.36)

        
        screen.add_widget(label)
        screen.add_widget(img)
        screen.add_widget(btn_flat)
        return screen

    def close_dialog(self,obj):
        self.dialog.dismiss()

        
    def start_dialog(self,obj):
        mask_detect()
        

if __name__=="__main__":
    Mask_Detection().run() 