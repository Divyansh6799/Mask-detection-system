from kivymd.app import MDApp
from kivy.app import App
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
                                       pos_hint={'center_x':0.50,'center_y':0.12},
                                       md_bg_color=self.theme_cls.primary_light,
                                       on_release=self.show_data)
        # icon_label=MDIcon(icon='video',halign='center',
        #                   pos_hint={'center_x':0.65,'center_y':0.12},
        #                   text_color=self.theme_cls.primary_color,
        #                   theme_text_color="Custom",
        #                 font_size= "48sp")
        # top_label=MDIcon(icon='youtube',halign='center',
        #                   pos_hint={'center_x':0.1,'center_y':0.92},
        #                 font_size="200sp")

        
        img=Image(source='background.jpg',opacity=.36)

        
        screen.add_widget(label)
        screen.add_widget(img)
        screen.add_widget(btn_flat)
        # screen.add_widget(top_label)
        # screen.add_widget(icon_label)
        
        return screen

    def show_data(self,obj):
        close_btn=MDFlatButton(text='Close',on_release=self.close_dialog)
        start_btn=MDFlatButton(text='Start',on_release=self.start_dialog)
        self.dialog=MDDialog(title='Important',
                             text='Click on Start to get started \npress q to exit after starting',
                             size_hint=(0.7,1),
                             buttons=[close_btn,start_btn])
        
        self.dialog.open()

        
    def close_dialog(self,obj):
        self.dialog.dismiss()

        
    def start_dialog(self,obj):
        mask_detect()

if __name__=="__main__":
    Mask_Detection().run() 