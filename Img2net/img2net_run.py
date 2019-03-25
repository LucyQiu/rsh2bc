#!/usr/bin/env python2

##################################################################### imports

import os
import gtk
import multiprocessing

import img2net_calc

##################################################################### gui

class img2net_class:

    def __init__(self):
        self.builder = gtk.Builder()        
        self.builder.add_from_file('gui.glade')
        self.window = self.builder.get_object('window1')
        self.builder.connect_signals(self)
        self.window.show()

    def on_window1_destroy(self,object,data=None):
        print('Closing img2net')
        gtk.main_quit()


    def on_combobox1_changed(self,object):
        model=self.builder.get_object('combobox1').get_model()
        index=self.builder.get_object('combobox1').get_active()
        gridtype = model[index][0]
        if(gridtype!='rectangular'):
            self.builder.get_object('combobox2').set_active(0)
            self.builder.get_object('combobox2').set_button_sensitivity(gtk.SENSITIVITY_OFF)
        else:
            self.builder.get_object('combobox2').set_button_sensitivity(gtk.SENSITIVITY_ON)

    def on_button1_clicked(self,object):

        print('Running img2net')

        dir_input=self.builder.get_object('filechooserbutton1').get_current_folder()
        dir_input=str(dir_input).replace('\\','/')

        model=self.builder.get_object('combobox1').get_model()
        index=self.builder.get_object('combobox1').get_active()
        gridtype = model[index][0]

        model=self.builder.get_object('combobox2').get_model()
        index=self.builder.get_object('combobox2').get_active()
        nullmodel = model[index][0]

        R=int(self.builder.get_object('spinbutton8').get_value())
        lz=int(self.builder.get_object('spinbutton1').get_value())

        dx=int(self.builder.get_object('spinbutton2').get_value())
        dy=int(self.builder.get_object('spinbutton3').get_value())
        dz=int(self.builder.get_object('spinbutton4').get_value())

        pbx=int(self.builder.get_object('checkbutton1').get_active())
        pby=int(self.builder.get_object('checkbutton2').get_active())
        pbz=int(self.builder.get_object('checkbutton3').get_active())

        vax=int(self.builder.get_object('spinbutton5').get_value())
        vay=int(self.builder.get_object('spinbutton6').get_value())

        cores=int(self.builder.get_object('spinbutton7').get_value())

        img2net_calc.calc(self,gtk,dir_input,gridtype,nullmodel,R,lz,dx,dy,dz,pbx,pby,pbz,vax,vay,cores)

##################################################################### run

if __name__ == '__main__':

    multiprocessing.freeze_support()

    app = img2net_class()

    gtk.main()
