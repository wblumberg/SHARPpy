import numpy as np
import sharppy.sharptab as tab
from sharppy.sharptab.constants import *
from sharppy.sharptab.profile import Profile, create_profile
from sharppy.viz.draggable import Draggable
from sharppy.viz.barbs import drawBarb
from PySide2 import QtGui, QtCore, QtWidgets
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtOpenGL import *
from utils.utils import total_seconds
import logging

from datetime import datetime, timedelta

__all__ = ['backgroundSkewT', 'plotSkewT']

class backgroundSkewT(QtWidgets.QWidget):
    def __init__(self, plot_omega=False):
        super(backgroundSkewT, self).__init__()
        self.plot_omega = plot_omega
        self.initUI()

    def initUI(self):
        '''
        Initialize the User Interface.

        '''
        logging.debug("Initalizing the backgroundSkewT.")

        self.lpad = 30; self.rpad = 65
        self.tpad = 20; self.bpad = 20
        self.tlx = self.rpad; self.tly = self.tpad
        self.wid = self.size().width() - self.rpad
        self.hgt = self.size().height() - self.bpad
        self.brx = self.wid ; self.bry = self.hgt
        self.pmax = 1050.; self.pmin = 100.
        self.barbx = self.brx + self.rpad / 2
        self.log_pmax = np.log(self.pmax); self.log_pmin = np.log(self.pmin)
        self.bltmpc = -50; self.brtmpc = 50; self.dt = 10
        self.xskew = 100 / 3.
        self.xrange = self.brtmpc - self.bltmpc
        self.yrange = np.tan(np.deg2rad(self.xskew)) * self.xrange
        self.clip = QRect(QPoint(self.lpad, self.tly), QPoint(self.brx + self.rpad, self.bry))
        self.originx = 0. # self.size().width() / 2
        self.originy = 0. # self.size().height() / 2
        self.scale = 1.
        if self.physicalDpiX() > 75:
            fsize = 6
            fsizet = 10
        else:
            fsize = 7
            fsizet = 14
        #fsize = np.floor(.06 * self.hgt)
        #fsizet = np.floor(.06 * self.hgt)

        self.title_font = QtGui.QFont('Arial', fsizet + (self.hgt * 0.01))
        self.title_metrics = QtGui.QFontMetrics( self.title_font )
        self.title_height = self.title_metrics.xHeight() + 5 + (self.hgt * 0.03)

        self.label_size =  fsize + 2 + (self.hgt * 0.0045)
        self.label_font = QtGui.QFont('Helvetica', self.label_size)
        self.env_trace_fsize = 11 + (self.hgt * 0.0045)
        self.environment_trace_font = QtGui.QFont('Helvetica', self.env_trace_fsize)
        self.in_plot_font = QtGui.QFont('Helvetica', fsize + (self.hgt * 0.0045))
        self.esrh_font = QtGui.QFont('Helvetica', fsize + 2 + (self.hgt * 0.0045))
        self.hght_font = QtGui.QFont('Helvetica', 9 + (self.hgt * 0.0045))

        self.esrh_metrics = QtGui.QFontMetrics( self.esrh_font )
        self.esrh_height = self.esrh_metrics.xHeight() + (self.hgt * 0.0045)
        self.plotBitMap = QtGui.QPixmap(self.width(), self.height())
        self.saveBitMap = None
        self.plotBitMap.fill(self.bg_color)
        self.plotBackground()
    
    def plotBackground(self):
        logging.debug("Plotting the background of skew.py")
        qp = QtGui.QPainter()
        qp.begin(self.plotBitMap)
        qp.setClipRect(self.clip)

        qp.translate(self.originx, self.originy)
        qp.scale(1. / self.scale, 1. / self.scale)

        self.transform = qp.transform()

        qp.scale(self.scale, self.scale)
        qp.translate(-self.originx, -self.originy)

        qp.setRenderHint(qp.Antialiasing)
        qp.setRenderHint(qp.TextAntialiasing)
        self.draw_frame(qp)
        for t in np.arange(self.bltmpc-100, self.brtmpc+self.dt, self.dt):
            self.draw_isotherm(t, qp)
        #for tw in range(self.bltmpc, self.brtmpc, 10): self.draw_moist_adiabat(tw, qp)
        for theta in np.arange(self.bltmpc, 80, 20): self.draw_dry_adiabat(theta, qp)
        for w in [2] + np.arange(4, 33, 4): self.draw_mixing_ratios(w, 600, qp)
        for p in [1000, 850, 700, 500, 300, 200, 100]:
            self.draw_isobar(p, 1, qp)
        for t in np.arange(self.bltmpc, self.brtmpc+self.dt, self.dt):
            self.draw_isotherm_labels(t, qp)
        for p in xrange(int(self.pmax), int(self.pmin-50), -50):
            self.draw_isobar(p, 0, qp)

        qp.end()
        self.backgroundBitMap = self.plotBitMap.copy(0, 0, self.width(), self.height())

    def resizeEvent(self, e):
        '''
        Resize the plot based on adjusting the main window.

        '''
        self.initUI()
    
    def wheelEvent(self, e):
        centerx, centery = e.x(), e.y()

        # From map
        max_speed = 100.
        delta = max(min(-e.delta(), max_speed), -max_speed)
        scale_fac = 10 ** (delta / 1000.)

        if self.scale * scale_fac > 1.0:
            scale_fac = 1. / self.scale

        self.scale *= scale_fac

        self.originx = centerx - (centerx - self.originx) / scale_fac
        self.originy = centery - (centery - self.originy) / scale_fac

        ll_x = self.originx + self.width() / self.scale
        ll_y = self.originy + self.height() / self.scale

        if ll_x < self.width():
            self.originx += (self.width() - ll_x)
        elif self.originx > 0:
            self.originx = 0

        if ll_y < self.height():
            self.originy += (self.height() - ll_y)
        elif self.originy > 0:
            self.originy = 0

        self.plotBackground()
        self.update()

    def draw_dry_adiabat(self, theta, qp):
        '''
        Draw the given moist adiabat.
        '''
        logging.debug("Drawing dry adiabat: " + str(theta))
        qp.setClipping(True)
        pen = QtGui.QPen(self.adiab_color, 1)
        pen.setStyle(QtCore.Qt.SolidLine)
        qp.setPen(pen)
        dp = -10
        presvals = np.arange(int(self.pmax), int(self.pmin)+dp, dp)
        thetas = ((theta + ZEROCNK) / (np.power((1000. / presvals),ROCP))) - ZEROCNK
        xvals = self.originx + self.tmpc_to_pix(thetas, presvals) / self.scale
        yvals = self.originy + self.pres_to_pix(presvals) / self.scale
        path = QPainterPath()
        path.moveTo(xvals[0], yvals[0])
        for i in xrange(1, len(presvals) ):
            p = presvals[i]
            x = xvals[i]
            y = yvals[i]
            path.lineTo(x, y)
        qp.drawPath(path)

    def draw_moist_adiabat(self, tw, qp):
        '''
        Draw the given moist adiabat.

        '''
        logging.debug("Drawing moist adiabats.")
        pen = QtGui.QPen(QtGui.QColor("#663333"), 1)
        pen.setStyle(QtCore.Qt.SolidLine)
        qp.setPen(pen)
        dp = -10
        for p in xrange(int(self.pmax), int(self.pmin)+dp, dp):
            t = tab.thermo.wetlift(1000., tw, p)
            x = self.tmpc_to_pix(t, p)
            y = self.pres_to_pix(p)
            if p == self.pmax:
                x2 = x; y2 = y
            else:
                x1 = x2; y1 = y2
                x2 = x; y2 = y
                qp.drawLine(x1, y1, x2, y2)

    def draw_mixing_ratios(self, w, pmin, qp):
        '''
        Draw the mixing ratios.

        '''
        logging.debug("Draw the water vapor mixing ratio line: " + str(w))
        qp.setClipping(True)
        t = tab.thermo.temp_at_mixrat(w, self.pmax)
        x1 = self.originx + self.tmpc_to_pix(t, self.pmax) / self.scale
        y1 = self.originy + self.pres_to_pix(self.pmax) / self.scale
        t = tab.thermo.temp_at_mixrat(w, pmin)
        x2 = self.originx + self.tmpc_to_pix(t, pmin) / self.scale
        y2 = self.originy + self.pres_to_pix(pmin) / self.scale
        rectF = QtCore.QRectF(x2-5, y2-10, 10, 10)
        pen = QtGui.QPen(self.bg_color, 1, QtCore.Qt.SolidLine)
        brush = QtGui.QBrush(self.bg_color, QtCore.Qt.SolidPattern)
        qp.setPen(pen)
        qp.setBrush(brush)
        qp.drawRect(rectF)
        pen = QtGui.QPen(self.mixr_color, 1, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.setFont(self.in_plot_font)
        qp.drawLine(x1, y1, x2, y2)
        qp.drawText(rectF, QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter,
            tab.utils.INT2STR(w))

    def draw_frame(self, qp):
        '''
        Draw the frame around the Skew-T.

        '''
        logging.debug("Drawing frame around the Skew-T.")
        qp.setClipping(False)
        pen = QtGui.QPen(self.bg_color, 0, QtCore.Qt.SolidLine)
        brush = QtGui.QBrush(self.bg_color, QtCore.Qt.SolidPattern)
        qp.setPen(pen)
        qp.setBrush(brush)
        qp.drawRect(0, 0, self.lpad, self.bry)
        qp.drawRect(0, self.pres_to_pix(self.pmax), self.brx, self.bry)
        qp.drawRect(self.brx, 0, self.wid+self.rpad,
                    self.pres_to_pix(self.pmax))
        pen = QtGui.QPen(self.fg_color, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(self.lpad, self.tpad, self.brx+self.rpad, self.tpad)
        qp.drawLine(self.brx+self.rpad, self.tpad, self.brx+self.rpad,
                    self.bry)
        qp.drawLine(self.brx+self.rpad, self.bry, self.lpad, self.bry)
        qp.drawLine(self.lpad, self.bry, self.lpad, self.tpad)

    def draw_isotherm_labels(self, t, qp):
        '''
        Add Isotherm Labels.

        '''
        logging.debug("Drawing isotherm label: " + str(t))
        pen = QtGui.QPen(self.fg_color)
        self.label_font.setBold(True)
        qp.setFont(self.label_font)
        x1 = self.originx + self.tmpc_to_pix(t, self.pmax) / self.scale

        if x1 >= self.lpad and x1 <= self.wid:
            qp.setClipping(False)
            qp.drawText(x1-10, self.bry+2, 20, 20,
                        QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter, tab.utils.INT2STR(t))
        self.label_font.setBold(False)

    def draw_isotherm(self, t, qp):
        '''
        Draw background isotherms.

        '''
        logging.debug("Drawing background isotherms: " + str(t))

        qp.setClipping(True)
        x1 = self.originx + self.tmpc_to_pix(t, self.pmax) / self.scale
        x2 = self.originx + self.tmpc_to_pix(t, self.pmin) / self.scale
        y1 = self.originy + self.bry / self.scale
        y2 = self.originy + self.tpad / self.scale
        if int(t) in [0, -20]:
            pen = QtGui.QPen(self.isotherm_hgz_color, 1)
        else:
            pen = QtGui.QPen(self.isotherm_color, 1)
        pen.setStyle(QtCore.Qt.CustomDashLine)
        pen.setDashPattern([4, 2])
        qp.setPen(pen)
        qp.drawLine(x1, y1, x2, y2)

    def draw_isobar(self, p, flag, qp):
        '''
        Draw background isobars.

        '''
        logging.debug("Drawing background isobar: " + str(p) + ' flag: ' + str(flag))
        pen = QtGui.QPen(self.fg_color, 1, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        self.label_font.setBold(True)
        qp.setFont(self.label_font)
        y1 = self.originy + self.pres_to_pix(p) / self.scale
        if y1 >= self.tpad and y1 <= self.hgt:
            offset = 5
            if flag:
                qp.drawLine(self.lpad, y1, self.brx, y1)
                qp.drawText(1, y1-20, self.lpad-4, 40,
                            QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight,
                            tab.utils.INT2STR(p))
            else:
                qp.drawLine(self.lpad, y1, self.lpad+offset, y1)
                qp.drawLine(self.brx+self.rpad-offset, y1,
                            self.brx+self.rpad, y1)
        self.label_font.setBold(False)

    def tmpc_to_pix(self, t, p):
        '''
        Function to convert a (temperature, pressure) coordinate
        to an X pixel.

        '''
        scl1 = self.brtmpc - (((self.bry - self.pres_to_pix(p)) /
                              (self.bry - self.tpad)) * self.yrange)
        return self.brx - (((scl1 - t) / self.xrange) * (self.brx - self.lpad))

    def pix_to_tmpc(self, x, y):
        '''
        Function to convert an (x, y) pixel into a temperature
        '''
        scl1 = self.brtmpc - (((self.bry - y) /
                              float(self.bry - self.tpad)) * self.yrange)
        return scl1 - (((self.brx - x) / float(self.brx - self.lpad)) * self.xrange)

    def pres_to_pix(self, p):
        '''
        Function to convert a pressure value (level) to a Y pixel.

        '''
        scl1 = self.log_pmax - self.log_pmin
        scl2 = self.log_pmax - np.log(p)
        return self.bry - (scl2 / scl1) * (self.bry - self.tpad)

    def pix_to_pres(self, y):
        '''
        Function to convert a Y pixel to a pressure level.

        '''
        scl1 = np.log(self.pmax) - np.log(self.pmin)
        scl2 = self.bry - float(y)
        scl3 = self.bry - self.tly + 1
        return self.pmax / np.exp((scl2 / scl3) * scl1)




class plotSkewT(backgroundSkewT):
    modified = Signal(int, dict)
    cursor_toggle = Signal(bool)
    cursor_move = Signal(float)
    parcel = Signal(tab.params.Parcel)
    reset = Signal(list)

    def __init__(self, **kwargs):
        logging.debug("Initializing plotSkewT.")
        self.bg_color = QtGui.QColor(kwargs.get('bg_color', '#000000'))
        self.fg_color = QtGui.QColor(kwargs.get('fg_color', '#FFFFFF'))
        self.isotherm_color = QtGui.QColor(kwargs.get('isotherm_color', '#555555'))
        self.isotherm_hgz_color = QtGui.QColor(kwargs.get('isotherm_hgz_color', '#0000FF'))
        self.adiab_color = QtGui.QColor(kwargs.get('adiab_color', '#333333'))
        self.mixr_color = QtGui.QColor(kwargs.get('mixr_color', '#006600'))

        super(plotSkewT, self).__init__(plot_omega=False)
        ## get the profile data
        self.prof = None
        self.prof_collections = []
        self.pc_idx = 0
        self.pcl = None

        self.all_observed = False
        self.plotdgz = kwargs.get('dgz', False)
        # PBL marker plotting functionality added by Nickolai Reimer NWS Billings, MT
        self.plotpbl = kwargs.get('pbl', False)
        self.interpWinds = kwargs.get('interpWinds', True)

        ## ui stuff
        self.title = kwargs.get('title', '')
        self.dp = -25
        self.temp_color = QtGui.QColor(kwargs.get('temp_color', '#FF0000'))
        self.ens_temp_color = QtGui.QColor(kwargs.get('ens_temp_color', '#880000'))
        self.dewp_color = QtGui.QColor(kwargs.get('dewp_color', '#00FF00'))
        self.ens_dewp_color = QtGui.QColor(kwargs.get('ens_dewp_color', '#008800'))
        self.wetbulb_color = QtGui.QColor(kwargs.get('wetbulb_color', '#00FFFF'))
        self.eff_layer_color = QtGui.QColor(kwargs.get('eff_layer_color', '#00FFFF'))
        self.background_colors =[ QtGui.QColor(c) for c in kwargs.get('background_colors', ['#6666CC', '#CC9966', '#66CC99']) ]

        self.hgt_color = QtGui.QColor(kwargs.get('hgt_color', '#FF0000'))
        self.dgz_color = QtGui.QColor(kwargs.get('dgz_color', '#F5D800'))

        self.lcl_mkr_color = QtGui.QColor(kwargs.get('lcl_mkr_color', '#00FF00'))
        self.lfc_mkr_color = QtGui.QColor(kwargs.get('lfc_mkr_color', '#FFFF00'))
        self.el_mkr_color = QtGui.QColor(kwargs.get('el_mkr_color', '#FF00FF'))

        self.sfc_units = kwargs.get('sfc_units', 'Fahrenheit')
        self.wind_units = kwargs.get('wind_units', 'knots')
        self.use_left = False
        self.setMouseTracking(True)
        self.was_right_click = False
        self.track_cursor = False
        self.readout = False
        self.readout_pres = 1000.
        self.cursor_loc = None
        self.drag_tmpc = None
        self.drag_dwpc = None
        ## create the readout labels
        self.presReadout = QLabel(parent=self)
        self.hghtReadout = QLabel(parent=self)
        self.tmpcReadout = QLabel(parent=self)
        self.dwpcReadout = QLabel(parent=self)
        ## set their alignments
        self.presReadout.setAlignment(QtCore.Qt.AlignCenter)
        self.hghtReadout.setAlignment(QtCore.Qt.AlignCenter)
        self.tmpcReadout.setAlignment(QtCore.Qt.AlignCenter)
        self.dwpcReadout.setAlignment(QtCore.Qt.AlignCenter)
        ## initialize the width to 0 so that they don't show up
        ## on initialization
        self.presReadout.setFixedWidth(0)
        self.hghtReadout.setFixedWidth(0)
        self.tmpcReadout.setFixedWidth(0)
        self.dwpcReadout.setFixedWidth(0)
        ## set the style sheet for text size, color, etc
        ## There's something funky going on with the colors here.
        fg_hex = "#%02x%02x%02x" % (self.bg_color.red(), self.bg_color.green(), self.bg_color.blue())
        bg_rgb = self.fg_color.getRgb()
#       print bg_rgb, self.fg_color.getRgb()
        rgb_string = 'rgb(' + str(bg_rgb[0]) + ',' + str(bg_rgb[1]) + ',' + str(bg_rgb[2]) + ',50%)'
        self.presReadout.setStyleSheet("QLabel {"
            "  background-color: " + rgb_string + ";"
            "  border-width: 0px;"
            "  font-size: 11px;"
            "  color: " + fg_hex + ";}")
        self.hghtReadout.setStyleSheet("QLabel {"
            "  background-color: " + rgb_string + ";"
            "  border-width: 0px;"
            "  font-size: 11px;"
            "  color: #FF0000;}")
        self.tmpcReadout.setStyleSheet("QLabel {"
            "  background-color: " + rgb_string + ";"
            "  border-width: 0px;"
            "  font-size: 11px;"
            "  color: #FF0000;}")
        self.dwpcReadout.setStyleSheet("QLabel {"
            "  background-color: " + rgb_string + ";"
            "  border-width: 0px;"
            "  font-size: 11px;"
            "  color: #00FF00;}")
        self.rubberBand = QRubberBand(QRubberBand.Line, self)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showCursorMenu)
        self.parcelmenu = QMenu("Lift Parcel")
        #ag = QtGui.QActionGroup(self, exclusive=True)

        # List of parcels that can be lifted
        pcl1 = QAction(self)
        pcl1.setText("This Parcel")
        pcl1.triggered.connect(lambda: self.liftparcellevel(0))
        #ag.addAction(pcl1)
        self.parcelmenu.addAction(pcl1)

        pcl2 = QAction(self)
        pcl2.setText("50 mb Layer Parcel")
        pcl2.triggered.connect(lambda: self.liftparcellevel(50))
        #ag.addAction(pcl2)
        self.parcelmenu.addAction(pcl2)

        pcl3 = QAction(self)
        pcl3.setText("100 mb Layer Parcel")
        pcl3.triggered.connect(lambda: self.liftparcellevel(100))
        #ag.addAction(pcl3)
        self.parcelmenu.addAction(pcl3)

        pcl4 = QAction(self)
        pcl4.setText("Custom Parcel")
        pcl4.triggered.connect(lambda: self.liftparcellevel(-9999))
        #ag.addAction(pcl4)
        self.parcelmenu.addAction(pcl4)
        self.parcelmenu.setEnabled(False)
        self.popupmenu=QMenu("Cursor Type:")
        ag = QActionGroup(self, exclusive=True)

        nocurs = QAction(self)
        nocurs.setText("No Cursor")
        nocurs.setCheckable(True)
        nocurs.setChecked(True)
        nocurs.triggered.connect(self.setNoCursor)
        a = ag.addAction(nocurs)
        self.popupmenu.addAction(a)

        storm_motion = QAction(self)
        storm_motion.setText("Readout Cursor")
        storm_motion.setCheckable(True)
        storm_motion.triggered.connect(self.setReadoutCursor)
        a = ag.addAction(storm_motion)
        self.popupmenu.addAction(a)

        self.popupmenu.addSeparator()
        self.popupmenu.addMenu(self.parcelmenu)

        #modify_sfc = QAction(self)
        #modify_sfc.setText("Modify Surface")
        #modify_sfc.setCheckable(True)
        #modify_sfc.setEnabled(False)
        #modify_sfc.triggered.connect(self.setReadoutCursor)
        #self.popupmenu.addAction(mod
        self.popupmenu.addSeparator()

        reset = QAction(self)
        reset.setText("Reset Skew-T")
        reset.triggered.connect(lambda: self.reset.emit(['tmpc', 'dwpc']))
        self.popupmenu.addAction(reset)

    def getPlotTitle(self, prof_coll):
        logging.debug("Calling getPlotTitle")

        modified = prof_coll.isModified() or prof_coll.isInterpolated()
        modified_str = "; Modified" if modified else ""

        loc = prof_coll.getMeta('loc')
        date = prof_coll.getCurrentDate()
        run = prof_coll.getMeta('run').strftime("%HZ")
        model = prof_coll.getMeta('model')
        observed = prof_coll.getMeta('observed')

        plot_title = loc + '   ' + datetime.strftime(date, "%Y%m%d/%H%M")
        if model == "Archive":
            fhour_str = ""
            if not prof_coll.getMeta('observed'):
                fhour = int(total_seconds(date - prof_coll.getMeta('base_time')) / 3600)
                fhour_str = " F%03d" % fhour
            plot_title += "  (User Selected" + fhour_str + modified_str + ")"
        elif model == "Analog":
            date = prof_coll.getAnalogDate()
            plot_title = loc + '   ' + datetime.strftime(date, "%Y%m%d/%H%M")
            plot_title += "  (Analog" + modified_str + ")"
        elif observed:
            plot_title += "  (Observed" + modified_str + ")"
        else:
            fhour = int(total_seconds(date - prof_coll.getMeta('base_time')) / 3600)
            plot_title += "  (" + run + "  " + model + "  " + ("F%03d" % fhour) + modified_str + ")"
        return plot_title

    def liftparcellevel(self, i):
        pres = self.pix_to_pres( self.cursor_loc.y())
        tmp = tab.interp.temp(self.prof, pres)
        dwp = tab.interp.dwpt(self.prof, pres)
        if i == 0:
            usrpcl = tab.params.parcelx(self.prof, flag=5, pres=pres, tmpc=tmp, dwpc=dwp)
        else:
            if i == -9999:
                depth, result = QInputDialog.getText(None, "Parcel Depth (" + str(int(pres)) + "to __)",\
                                            "Mean Layer Depth (mb):")
                try:
                    i = int(depth)
                except:
                    return
            user_initpcl = tab.params.DefineParcel(self.prof, flag=4, pbot=pres, pres=i)
            usrpcl = tab.params.parcelx(self.prof, pres=user_initpcl.pres, tmpc=user_initpcl.tmpc,\
                                          dwpc=user_initpcl.dwpc)
        self.parcel.emit(usrpcl) # Emit a signal that a new profile has been created

    def addProfileCollection(self, prof_coll):
        logging.debug("Adding profile collection:" + str(prof_coll))
        self.prof_collections.append(prof_coll)

    def rmProfileCollection(self, prof_coll):
        logging.debug("Removing profile collection:" + str(prof_coll))
        self.prof_collections.remove(prof_coll)

    def setActiveCollection(self, pc_idx, **kwargs):
        logging.debug("Setting the active collection to the Skew-T...")
        self.pc_idx = pc_idx
        prof = self.prof_collections[pc_idx].getHighlightedProf()
        self.plot_omega = not self.prof_collections[pc_idx].getMeta('observed')
        self.prof = prof

        self.pres = prof.pres; self.hght = prof.hght
        self.tmpc = prof.tmpc; self.dwpc = prof.dwpc
        self.vtmp = prof.vtmp
        self.dew_stdev = prof.dew_stdev
        self.tmp_stdev = prof.tmp_stdev
        self.u = prof.u; self.v = prof.v
        self.wetbulb = prof.wetbulb
        self.interpWinds = kwargs.get('interpWinds', True)

        trans_tmx = self.originx + self.tmpc_to_pix(self.tmpc, self.pres) / self.scale
        trans_dwx = self.originx + self.tmpc_to_pix(self.dwpc, self.pres) / self.scale
        trans_y = self.originy + self.pres_to_pix(self.pres) / self.scale

        self.drag_tmpc = Draggable(trans_tmx, trans_y, self.plotBitMap, lock_dim='y', line_color=QtGui.QColor("#9F0101"))
        self.drag_dwpc = Draggable(trans_dwx, trans_y, self.plotBitMap, lock_dim='y', line_color=QtGui.QColor("#019B06"))

        if kwargs.get('update_gui', True):
            self.clearData()
            self.plotData()
            if self.readout:
                self.updateReadout()
            self.update()

    def setParcel(self, parcel):
        logging.debug("Setting the parcel: " + str(parcel))
        self.pcl = parcel

        self.clearData()
        self.plotData()
        if self.readout:
            self.updateReadout()
        self.update()

    def setDGZ(self, flag):
        logging.debug("PlotDGZ Flag: " + str(flag))
        self.plotdgz = flag

        self.clearData()
        self.plotData()
        self.update()
        return

    def setPBLLevel(self, flag):
        self.plotpbl = flag
        
        self.clearData()
        self.plotData()
        self.update()
        return

    def setAllObserved(self, all_observed, update_gui=True):
        self.all_observed = all_observed

        if update_gui:
            self.clearData()
            self.plotData()
            self.update()

    def setPreferences(self, update_gui=True, **kwargs):
        self.bg_color = QtGui.QColor(kwargs['bg_color'])
        self.fg_color = QtGui.QColor(kwargs['fg_color'])
        self.isotherm_color = QtGui.QColor(kwargs['skew_itherm_color'])
        self.isotherm_hgz_color = QtGui.QColor(kwargs['skew_itherm_hgz_color'])
        self.adiab_color = QtGui.QColor(kwargs['skew_adiab_color'])
        self.mixr_color = QtGui.QColor(kwargs['skew_mixr_color'])

        self.temp_color = QtGui.QColor(kwargs['temp_color'])
        self.dewp_color = QtGui.QColor(kwargs['dewp_color'])
        self.wetbulb_color = QtGui.QColor(kwargs['wetb_color'])
        self.eff_layer_color = QtGui.QColor(kwargs['eff_inflow_color'])
        self.hgt_color = QtGui.QColor(kwargs['skew_hgt_color'])
        self.dgz_color = QtGui.QColor(kwargs['skew_dgz_color'])

        self.lcl_mkr_color = QtGui.QColor(kwargs['skew_lcl_mkr_color'])
        self.lfc_mkr_color = QtGui.QColor(kwargs['skew_lfc_mkr_color'])
        self.el_mkr_color = QtGui.QColor(kwargs['skew_el_mkr_color'])

        self.sfc_units = kwargs['temp_units']
        self.wind_units = kwargs['wind_units']

        if update_gui:
            self.plotBitMap.fill(self.bg_color)
            self.plotBackground()
            self.clearData()
            self.plotData()
            self.update()

    def setDeviant(self, deviant):
        self.use_left = deviant == 'left'

        self.clearData()
        self.plotData()
        self.update()

    def _restTmpc(self, x, y):
        xs, ys = self.drag_dwpc.getCoords()
        idx = np.argmin(np.abs(y - ys))
        return max(xs[idx], x),  y

    def _restDwpc(self, x, y):
        xs, ys = self.drag_tmpc.getCoords()
        idx = np.argmin(np.abs(y - ys))
        return min(xs[idx], x),  y

    def mouseReleaseEvent(self, e):
        logging.debug("Releasing the mouse in skew-T.")
        if self.prof is None:
            return

        if not self.was_right_click and self.readout:
            self.track_cursor = not self.track_cursor
            self.cursor_loc = e.pos()

        drag_idx = None
        if self.drag_tmpc.isDragging():
            drag_idx, rls_x, rls_y = self.drag_tmpc.release(e.x(), e.y(), restrictions=self._restTmpc)
            prof_name = 'tmpc'
        elif self.drag_dwpc.isDragging():
            drag_idx, rls_x, rls_y = self.drag_dwpc.release(e.x(), e.y(), restrictions=self._restDwpc)
            prof_name = 'dwpc'

        if drag_idx is not None:
            trans_x = (rls_x - self.originx) * self.scale
            trans_y = (rls_y - self.originy) * self.scale
            tmpc = self.pix_to_tmpc(trans_x, trans_y)

            self.modified.emit(drag_idx, {prof_name:tmpc})

        self.was_right_click = False

    def mousePressEvent(self, e):
        logging.debug("Pressing the mouse in the skew-T.")
        if self.prof is None:
            return

        self.was_right_click = e.button() & QtCore.Qt.RightButton

        if not self.was_right_click and not self.readout:
            drag_started = False
            for drag in [ self.drag_dwpc, self.drag_tmpc ]:
                if not drag_started:
                    drag_started = drag.click(e.x(), e.y())

    def mouseMoveEvent(self, e):
        if self.prof is None:
            return

        if self.track_cursor:
            self.readout_pres = self.pix_to_pres((e.y() - self.originy) * self.scale)
            self.updateReadout()

        for drag, rest in [ (self.drag_dwpc, self._restDwpc), (self.drag_tmpc, self._restTmpc) ]:
            drag.drag(e.x(), e.y(), restrictions=rest)

        self.update()

    def updateReadout(self):
        y = self.originy + self.pres_to_pix(self.readout_pres) / self.scale
        hgt = tab.interp.to_agl( self.prof, tab.interp.hght(self.prof, self.readout_pres) )
        tmp = tab.interp.temp(self.prof, self.readout_pres)
        dwp = tab.interp.dwpt(self.prof, self.readout_pres)
        has_omega = True
        try:
            omg = tab.interp.omeg(self.prof, self.readout_pres)
        except:
            has_omega = False

        thae = tab.interp.thetae(self.prof, self.readout_pres)
        tw = tab.interp.wetbulb(self.prof, self.readout_pres)
        tha = tab.interp.theta(self.prof, self.readout_pres)
        q = tab.interp.mixratio(self.prof, self.readout_pres)
        if has_omega:
            #omg = omg * 10 # converts to microbars/s (units on the default axis)
            omg = omg * 36 # converts to mb/hr (units used in Eric Thaler's QG solver)

        #print hgt, tmp, dwp, omg, thae, tw, tha, q
        thetae_unicode = u"\u03B8" + 'e='
        theta_unicode = u"\u03B8" + '='
        omega_unicode = u"\u03C9" + '='
        microbars_units = u"\u00B5" + 'b/s'
        self.rubberBand.setGeometry(QRect(QPoint(self.lpad,y), QPoint(self.brx,y)).normalized())
        self.presReadout.setFixedWidth(60)
        self.hghtReadout.setFixedWidth(65)
        self.tmpcReadout.setFixedWidth(65)
        self.dwpcReadout.setFixedWidth(65)
        self.presReadout.setText(tab.utils.FLOAT2STR(self.readout_pres, 1) + ' hPa')
        self.hghtReadout.setText(tab.utils.FLOAT2STR(hgt, 1) + ' m')
        self.tmpcReadout.setText(theta_unicode + tab.utils.FLOAT2STR(tha, 1) + ' K')
        if has_omega:
            self.dwpcReadout.setText(omega_unicode + tab.utils.FLOAT2STR(omg, 1) + ' mb/hr')
        else:
            self.dwpcReadout.setText(thetae_unicode + tab.utils.FLOAT2STR(thae, 1) + ' K')

        self.presReadout.move(self.lpad, y+2)
        self.hghtReadout.move(self.lpad, y - 15)
        self.tmpcReadout.move(self.brx-self.rpad, y - 15)
        self.dwpcReadout.move(self.brx-self.rpad, y+2)
        self.rubberBand.show()
        self.cursor_move.emit(hgt)

    def setReadoutCursor(self):
        logging.debug("Turning on the readout cursor.")
        self.parcelmenu.setEnabled(True)
        self.readout = True
        self.track_cursor = True
        self.presReadout.show()
        self.hghtReadout.show()
        self.tmpcReadout.show()
        self.dwpcReadout.show()
        self.rubberBand.show()
        self.cursor_toggle.emit(True)
        self.clearData()
        self.plotData()
        self.update()
        self.parentWidget().setFocus()

    def setNoCursor(self):
        logging.debug("Turning off the readout cursor.")
        self.parcelmenu.setEnabled(False)
        self.readout = False
        self.track_cursor = False
        self.presReadout.hide()
        self.hghtReadout.hide()
        self.tmpcReadout.hide()
        self.dwpcReadout.hide()      
        self.rubberBand.hide()
        self.cursor_toggle.emit(False)
        self.clearData()
        self.plotData()
        self.update()
        self.parentWidget().setFocus()

    def resizeEvent(self, e):
        '''
        Resize the plot based on adjusting the main window.

        '''
        logging.debug("Resizing the Skew-T")
        super(plotSkewT, self).resizeEvent(e)
        self.plotData()

    def closeEvent(self, e):
        pass

    def showCursorMenu(self, pos):
        logging.debug("Displaying the cursor menu.")
        if self.cursor_loc is None or self.track_cursor:
            self.cursor_loc = pos
        self.popupmenu.popup(self.mapToGlobal(pos))

    def wheelEvent(self, e):
        super(plotSkewT, self).wheelEvent(e)

        trans_tmx = self.originx + self.tmpc_to_pix(self.tmpc, self.pres) / self.scale
        trans_dwx = self.originx + self.tmpc_to_pix(self.dwpc, self.pres) / self.scale
        trans_y = self.originy + self.pres_to_pix(self.pres) / self.scale

        self.drag_tmpc.setCoords(trans_tmx, trans_y)
        self.drag_dwpc.setCoords(trans_dwx, trans_y)

        self.plotBitMap.fill(self.bg_color)
        if self.readout:
            self.updateReadout()
        self.plotBackground()
        self.plotData()

    def paintEvent(self, e):
        logging.debug("Calling paintEvent.")
        super(plotSkewT, self).paintEvent(e)
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.drawPixmap(0, 0, self.plotBitMap)
        qp.end()
    
    def clearData(self):
        '''
        Handles the clearing of the pixmap
        in the frame.
        '''
        logging.debug("Clearing the data from the Skew-T.")
        self.plotBitMap = self.backgroundBitMap.copy(0, 0, self.width(), self.height())
        for drag in [ self.drag_dwpc, self.drag_tmpc ]:
            if drag is not None:
                drag.setBackground(self.plotBitMap)

    def plotData(self):
        '''
        Plot the data used in a Skew-T.

        '''
        logging.debug("Plotting the data on the Skew-T")
        if self.prof is None:
            return

        qp = QtGui.QPainter()
        qp.begin(self.plotBitMap)
        qp.setClipRect(self.clip)

        qp.setRenderHint(qp.Antialiasing)
        qp.setRenderHint(qp.TextAntialiasing)
        self.drawTitles(qp)

        bg_color_idx = 0

        cur_dt = self.prof_collections[self.pc_idx].getCurrentDate()
        for idx, prof_col in enumerate(self.prof_collections):
            # Plot all unhighlighted members at this time
            if prof_col.getCurrentDate() == cur_dt:
                proflist = prof_col.getCurrentProfs().values()
                if idx == self.pc_idx:
                    temp_color = self.ens_temp_color
                    dewp_color = self.ens_dewp_color
                else:
                    temp_color = self.background_colors[bg_color_idx]
                    dewp_color = self.background_colors[bg_color_idx]

                    bg_color_idx = (bg_color_idx + 1) % len(self.background_colors)

                for profile in proflist:
                    self.drawTrace(profile.tmpc, temp_color, qp, p=profile.pres, width=1)
                    self.drawTrace(profile.dwpc, dewp_color, qp, p=profile.pres, width=1)
                    self.drawBarbs(profile, qp, color="#666666")

        bg_color_idx = 0
        for idx, prof_col in enumerate(self.prof_collections):
            if idx != self.pc_idx and (prof_col.getCurrentDate() == cur_dt or self.all_observed):
                profile = prof_col.getHighlightedProf()

                color = self.background_colors[bg_color_idx]

                self.drawTrace(profile.tmpc, color, qp, p=profile.pres)
                self.drawTrace(profile.dwpc, color, qp, p=profile.pres)
                self.drawBarbs(profile, qp, color=color)

                bg_color_idx = (bg_color_idx + 1) % len(self.background_colors)

        self.drawTrace(self.wetbulb, self.wetbulb_color, qp, width=1)
        self.drawTrace(self.tmpc, self.temp_color, qp, stdev=self.tmp_stdev)
        self.drawTrace(self.vtmp, self.temp_color, qp, width=1, style=QtCore.Qt.DashLine, label=False)

        if self.plotdgz is True and (self.prof.dgz_pbot != self.prof.dgz_ptop):
#           idx = np.ma.where((self.prof.pres <= self.prof.dgz_pbot) & (self.prof.pres >= self.prof.dgz_ptop))
#           idx_pbot, idx_ptop = np.searchsorted(np.array([self.prof.dgz_pbot, self.prof.dgz_ptop]), self.prof.pres)

            pres = np.ma.masked_invalid(np.arange(self.prof.dgz_ptop, self.prof.dgz_pbot, 5)[::-1])
            tmpc = np.ma.masked_invalid(tab.interp.temp(self.prof, pres))

            qp.setFont(self.hght_font)
            self.drawTrace(tmpc, QtGui.QColor("#F5D800"), qp, p=pres, label=False)
            self.draw_sig_levels(qp, plevel=self.prof.dgz_pbot, color="#F5D800")
            self.draw_sig_levels(qp, plevel=self.prof.dgz_ptop, color="#F5D800")

        self.drawTrace(self.dwpc, self.dewp_color, qp, stdev=self.dew_stdev)

        for h in [0,1000.,3000.,6000.,9000.,12000.,15000.]:
            self.draw_height(h, qp)
        if self.pcl is not None:
            self.dpcl_ttrace = self.prof.dpcl_ttrace
            self.dpcl_ptrace = self.prof.dpcl_ptrace
            self.drawVirtualParcelTrace(self.pcl.ttrace, self.pcl.ptrace, qp)
            self.drawVirtualParcelTrace(self.dpcl_ttrace, self.dpcl_ptrace, qp, color=QtGui.QColor("#FF00FF"))
        
        if self.plotpbl:
            self.draw_pbl_level(qp)
        
        self.draw_parcel_levels(qp)
        qp.setRenderHint(qp.Antialiasing, False)
        self.drawBarbs(self.prof, qp)
        qp.setRenderHint(qp.Antialiasing)

        self.draw_effective_layer(qp)
        if self.plot_omega:
            self.draw_omega_profile(qp)

        qp.end()

    def drawBarbs(self, prof, qp, color=None):
        logging.debug("Drawing the wind barbs on the Skew-T.")
        if color is None:
            color = self.fg_color

        qp.setClipping(False)

        rect_size = self.clip.size()
        rect_size.setHeight(rect_size.height() + self.bpad)
        mod_clip = QRect(self.clip.topLeft(), rect_size)
        qp.setClipRect(mod_clip)

        if self.interpWinds is False:
            i = 0
            mask1 = prof.u.mask
            mask2 = prof.pres.mask
            mask = np.maximum(mask1, mask2)
            pres = prof.pres[~mask]
            u = prof.u[~mask]
            v = prof.v[~mask]
            wdir, wspd = tab.utils.comp2vec(u, v)
            yvals = self.originy + self.pres_to_pix(pres) / self.scale
            for y in yvals:
                dd = wdir[i]
                ss = wspd[i]

                if self.wind_units == 'm/s':
                    ss = tab.utils.KTS2MS(ss)
                drawBarb( qp, self.barbx, y, dd, vv, shemis=(prof.latitude < 0) )
                i += 1
        else:
            pres = np.arange(prof.pres[prof.sfc], prof.pres[prof.top], -40)
            wdir, wspd = tab.interp.vec(prof, pres)

            for p, dd, ss in zip(pres, wdir, wspd):
                if not tab.utils.QC(dd) or np.isnan(ss) or p < self.pmin:
                    continue

                if self.wind_units == 'm/s':
                    ss = tab.utils.KTS2MS(ss)

                y = self.originy + self.pres_to_pix(p) / self.scale
                drawBarb( qp, self.barbx, y, dd, ss, color=color, shemis=(prof.latitude < 0) )
        qp.setClipRect(self.clip)

    def drawTitles(self, qp):
        logging.debug("Drawing the titles on the Skew-T")
        box_width = 150

        cur_dt = self.prof_collections[self.pc_idx].getCurrentDate()
        idxs, titles = zip(*[ (idx, self.getPlotTitle(pc)) for idx, pc in enumerate(self.prof_collections) if pc.getCurrentDate() == cur_dt or self.all_observed ])
        titles = list(titles)
        main_title = titles.pop(idxs.index(self.pc_idx))

        qp.setClipping(False)
        qp.setFont(self.title_font)

        pen = QtGui.QPen(self.fg_color, 1, QtCore.Qt.SolidLine)
        qp.setPen(pen)

        rect0 = QtCore.QRect(self.lpad, 2, box_width, self.title_height)
        qp.drawText(rect0, QtCore.Qt.TextDontClip | QtCore.Qt.AlignLeft, main_title)

        bg_color_idx = 0
        for idx, title in enumerate(titles):
            pen = QtGui.QPen(QtGui.QColor(self.background_colors[bg_color_idx]), 1, QtCore.Qt.SolidLine)
            qp.setPen(pen)

            rect0 = QtCore.QRect(self.width() - box_width, 2 + idx * self.title_height, box_width, self.title_height)
            qp.drawText(rect0, QtCore.Qt.TextDontClip | QtCore.Qt.AlignRight, title)

            bg_color_idx = (bg_color_idx + 1) % len(self.background_colors)

    def draw_height(self, h, qp):
        logging.debug("Drawing the height marker: " + str(h))
        qp.setClipping(True)
        pen = QtGui.QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.setFont(self.hght_font)
        offset = 10
        txt_offset = 15 
        sfc = tab.interp.hght( self.prof, self.prof.pres[self.prof.sfc] )
        p1 = tab.interp.pres(self.prof, h+sfc)
        if np.isnan(p1) == False:
            y1 = self.originy + self.pres_to_pix(p1) / self.scale
            qp.drawLine(self.lpad, y1, self.lpad+offset, y1)
            qp.drawText(self.lpad+txt_offset, y1-20, self.lpad+txt_offset, 40,
                QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                tab.utils.INT2STR(h/1000)+' km')

    def draw_sig_levels(self, qp, plevel=1000, color=None):
        logging.debug("Drawing singificant levels.")
        if color is None:
            color = self.fg_color

        qp.setClipping(True)
        if not tab.utils.QC(plevel):
            return
        xbounds = [37,45]
        z = tab.utils.M2FT(tab.interp.hght(self.prof, plevel))

        x = self.tmpc_to_pix(xbounds, [1000.,1000.])
        y = self.originy + self.pres_to_pix(plevel) / self.scale
        pen = QtGui.QPen(color)
        pen.setWidth(1)
        pen.setStyle(QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(x[0], y, x[1], y)
        rect1 = QtCore.QRectF(self.tmpc_to_pix(29, 1000.), y-3, x[1] - x[0], 4) 
        qp.drawText(rect1, QtCore.Qt.TextDontClip | QtCore.Qt.AlignLeft, tab.utils.INT2STR(z) + '\'')
         
    def draw_pbl_level(self, qp):
        logging.debug("Drawing the PBL top marker.")
        if self.prof is not None:
            qp.setClipping(True)
            xbounds = [37,41]
            x = self.tmpc_to_pix(xbounds, [1000.,1000.])
            pblp = self.prof.ppbl_top
            if tab.utils.QC(pblp):
                y = self.originy + self.pres_to_pix(pblp) / self.scale
                pen = QtGui.QPen(QtCore.Qt.gray, 2, QtCore.Qt.SolidLine)
                qp.setPen(pen)
                qp.drawLine(x[0], y, x[1], y)
                rect1 = QtCore.QRectF(x[0], y+6, x[1] - x[0], 4) 
                qp.drawText(rect1, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, "PBL")

    def draw_parcel_levels(self, qp):
        logging.debug("Drawing the parcel levels (LCL, LFC, EL).")
        if self.pcl is None:
            return
        qp.setClipping(True)
        xbounds = [37,41]
        x = self.tmpc_to_pix(xbounds, [1000.,1000.])
        lclp = self.pcl.lclpres
        lfcp = self.pcl.lfcpres
        elp = self.pcl.elpres

        # Plot LCL
        if tab.utils.QC(lclp):
            y = self.originy + self.pres_to_pix(lclp) / self.scale
            pen = QtGui.QPen(self.lcl_mkr_color, 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.setFont(self.hght_font)
            qp.drawLine(x[0], y, x[1], y)
            rect1 = QtCore.QRectF(x[0], y+6, x[1] - x[0], 4) 
            qp.drawText(rect1, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, "LCL")
        # Plot LFC
        if tab.utils.QC(lfcp):
            y = self.originy + self.pres_to_pix(lfcp) / self.scale
            pen = QtGui.QPen(self.lfc_mkr_color, 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.setFont(self.hght_font)
            qp.drawLine(x[0], y, x[1], y)
            rect2 = QtCore.QRectF(x[0], y-8, x[1] - x[0], 4) 
            qp.drawText(rect2, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, "LFC")
        # Plot EL
        if tab.utils.QC(elp) and elp != lclp:
            y = self.originy + self.pres_to_pix(elp) / self.scale
            pen = QtGui.QPen(self.el_mkr_color, 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.setFont(self.hght_font)
            qp.drawLine(x[0], y, x[1], y)
            rect3 = QtCore.QRectF(x[0], y-8, x[1] - x[0], 4) 
            qp.drawText(rect3, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, "EL")

    def omeg_to_pix(self, omeg):
        plus10_bound = -49
        minus10_bound = -41
        x1_m10 = self.tmpc_to_pix(minus10_bound, 1000)
        x1_10 = self.tmpc_to_pix(plus10_bound, 1000)
        x1_0 = self.tmpc_to_pix((plus10_bound + minus10_bound)/2., 1000)
        if omeg > 0:
            return ((x1_0 - x1_10)/(0.-10.)) * omeg + x1_0    
        elif omeg < 0:
            return ((x1_0 - x1_m10)/(0.+10.)) * omeg + x1_0 
        else:
            return x1_0

    def draw_omega_profile(self, qp):
        logging.debug("Drawing the omega profile.")
        qp.setClipping(True)
        plus10_bound = -49
        minus10_bound = -41

        x1_m10 = self.tmpc_to_pix(minus10_bound, 1000)
        y1_m10 = self.pres_to_pix(1000)
        y2_m10 = self.pres_to_pix(111)
        pen = QtGui.QPen(QtCore.Qt.magenta, 1, QtCore.Qt.DashDotLine)
        qp.setPen(pen)
        qp.drawLine(x1_m10, y1_m10, x1_m10, y2_m10)
        x1_10 = self.tmpc_to_pix(plus10_bound, 1000)
        y1_10 = self.pres_to_pix(1000)
        y2_10 = self.pres_to_pix(111)
        qp.drawLine(x1_10, y1_10, x1_10, y2_10)
        pen = QtGui.QPen(QtCore.Qt.magenta, 1, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        x1 = self.tmpc_to_pix((plus10_bound + minus10_bound)/2., 1000)
        y1 = self.pres_to_pix(1000)
        y2 = self.pres_to_pix(111)
        qp.drawLine(x1, y1, x1, y2)
        rect3 = QtCore.QRectF(x1_10, y2 - 18, x1_m10 - x1_10, 4) 
        qp.drawText(rect3, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, "OMEGA")  
        rect3 = QtCore.QRectF(x1_m10-3, y2 - 7, 5, 4) 
        qp.drawText(rect3, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, "-10")               
        rect3 = QtCore.QRectF(x1_10-3, y2 - 7, 5, 4) 
        qp.drawText(rect3, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, "+10")

        x1 = self.tmpc_to_pix((plus10_bound + minus10_bound)/2., 1000)

        for i in range(len(self.prof.omeg)):
            pres_y = self.originy + self.pres_to_pix(self.prof.pres[i]) / self.scale
            if not tab.utils.QC(self.prof.omeg[i]) or self.prof.pres[i] < 111:
                continue
            if self.prof.omeg[i] > 0:
                pen = QtGui.QPen(QtGui.QColor("#0066CC"), 1.5, QtCore.Qt.SolidLine)
            elif self.prof.omeg[i] < 0:
                pen = QtGui.QPen(QtGui.QColor("#FF6666"), 1.5, QtCore.Qt.SolidLine)
            else:
                pen = QtGui.QPen(QtCore.Qt.magenta, 1, QtCore.Qt.SolidLine)
            qp.setPen(pen)                
            x2 = self.omeg_to_pix(self.prof.omeg[i]*10.)
            qp.drawLine(x1, pres_y, x2, pres_y)


    def draw_effective_layer(self, qp):
        '''
        Draw the bounds of the effective inflow layer.
        '''
        logging.debug("Drawing the effective inflow layer.")
        qp.setClipping(True)
        ptop = self.prof.etop; pbot = self.prof.ebottom
        len = 15
        text_offset = 10
        if tab.utils.QC(ptop) and tab.utils.QC(pbot):
            x1 = self.tmpc_to_pix(-20, 1000)
            x2 = self.tmpc_to_pix(-33, 1000)
            y1 = self.originy + self.pres_to_pix(pbot) / self.scale
            y2 = self.originy + self.pres_to_pix(ptop) / self.scale
            rect1 = QtCore.QRectF(x2, y1-(self.esrh_height)/2., 25, self.esrh_height)
            rect2 = QtCore.QRectF(x2, y2-(self.esrh_height)/2., 50, self.esrh_height)
            rect3 = QtCore.QRectF(x1-15, y2-self.esrh_height-5, 50, self.esrh_height)
            pen = QtGui.QPen(self.bg_color, 0, QtCore.Qt.SolidLine)
            brush = QtGui.QBrush(self.bg_color, QtCore.Qt.SolidPattern)
            qp.setPen(pen)
            qp.setBrush(brush)
            sfc = tab.interp.hght( self.prof, self.prof.pres[self.prof.sfc] )
            if self.prof.pres[ self.prof.sfc ] == pbot:
                text_bot = 'SFC'
            else:
                text_bot = tab.interp.hght(self.prof, pbot) - sfc
                text_bot = tab.utils.INT2STR( text_bot ) + 'm'
            text_top = tab.interp.hght(self.prof, ptop) - sfc
            text_top = tab.utils.INT2STR( text_top ) + 'm'
            qp.drawRect(rect1)
            qp.drawRect(rect2)
            qp.drawRect(rect3)
            pen = QtGui.QPen(self.eff_layer_color, 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.setFont(self.esrh_font)
            qp.drawLine(x1-len, y1, x1+len, y1)
            qp.drawLine(x1-len, y2, x1+len, y2)
            qp.drawLine(x1, y1, x1, y2)
            qp.setClipping(False)
            qp.drawText(rect1, QtCore.Qt.TextDontClip | QtCore.Qt.AlignLeft, text_bot)
            qp.setClipping(True)
            qp.drawText(rect2, QtCore.Qt.TextDontClip | QtCore.Qt.AlignLeft, text_top)

            if self.use_left:
                esrh = self.prof.left_esrh[0]
            else:
                esrh = self.prof.right_esrh[0]

            qp.drawText(rect3, QtCore.Qt.TextDontClip | QtCore.Qt.AlignLeft,
                tab.utils.INT2STR(esrh) + ' m2s2')
           # qp.drawText(x1-2*len, y1-text_offset, 40, 40,
           #     QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight,
           #     text_bot)
    
    def drawVirtualParcelTrace(self, ttrace, ptrace, qp, width=1, color=None):
        '''
        Draw a parcel trace.
        '''
        logging.debug("Drawing the virtual parcel trace.")
        if color is None:
            color = self.fg_color

        qp.setClipping(True)
        pen = QtGui.QPen(color, width, QtCore.Qt.DashLine)
        brush = QtGui.QBrush(QtCore.Qt.NoBrush)
        qp.setPen(pen)
        qp.setBrush(brush)
        path = QPainterPath()
        if not tab.utils.QC(ptrace):
            return
        yvals = self.originy + self.pres_to_pix(ptrace) / self.scale
        xvals = self.originx + self.tmpc_to_pix(ttrace, ptrace) / self.scale
        path.moveTo(xvals[0], yvals[0])
        for i in xrange(1, len(yvals)):
            x = xvals[i]; y = yvals[i]
#           if y < self.tpad:
#               break
#           else:
            path.lineTo(x, y)
        qp.drawPath(path)

    def drawTrace(self, data, color, qp, width=3, style=QtCore.Qt.SolidLine, p=None, stdev=None, label=True):
        '''
        Draw an environmental trace.

        '''
        logging.debug("Drawing an environmental profile trace.")
        qp.setClipping(True)
        pen = QtGui.QPen(QtGui.QColor(color), width, style)
        pen.setWidth(width)
        brush = QtGui.QBrush(QtCore.Qt.NoBrush)
        qp.setPen(pen)
        qp.setBrush(brush)

        mask1 = data.mask
        if p is not None:
            mask2 = p.mask
            pres = p
        else:
            mask2 = self.pres.mask
            pres = self.pres
        mask = np.maximum(mask1, mask2)
        data = data[~mask]
        pres = pres[~mask]
        if stdev is not None:
            stdev = stdev[~mask]

        path = QPainterPath()
        x = self.originx + self.tmpc_to_pix(data, pres) / self.scale
        y = self.originy + self.pres_to_pix(pres) / self.scale

        path.moveTo(x[0], y[0])
        for i in xrange(1, x.shape[0]):
            path.lineTo(x[i], y[i])
            if stdev is not None:
                self.drawSTDEV(pres[i], data[i], stdev[i], color, qp)

        qp.drawPath(path)

        if label is True:
            qp.setClipping(False)
            if self.sfc_units == 'Celsius':
                label = data[0]
            else:
                label = tab.thermo.ctof(data[0]) #(1.8 * data[0]) + 32.
            pen = QtGui.QPen(self.bg_color, 0, QtCore.Qt.SolidLine)
            brush = QtGui.QBrush(self.bg_color, QtCore.Qt.SolidPattern)
            qp.setPen(pen)
            qp.setBrush(brush)
            rect = QtCore.QRectF(x[0]-8, y[0]+4, 16, 12)
            qp.drawRect(rect)
            pen = QtGui.QPen(QtGui.QColor(color), 3, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.setFont(self.environment_trace_font)
            qp.drawText(rect, QtCore.Qt.AlignCenter, tab.utils.INT2STR(label))
            qp.setClipping(False)

    def drawSTDEV(self, pres, data, stdev, color, qp, width=1):
        '''
        Draw the error bars on the profile.
        '''
        pen = QtGui.QPen(QtGui.QColor(color), width, QtCore.Qt.SolidLine)
        brush = QtGui.QBrush(QtCore.Qt.NoBrush)
        qp.setPen(pen)
        qp.setBrush(brush)
        path = QPainterPath()
        offset = 5.
        x = self.tmpc_to_pix(data, pres)
        y = self.pres_to_pix(pres)
        err_left = self.tmpc_to_pix(data - stdev, pres)
        err_right = self.tmpc_to_pix(data + stdev, pres)
        path.moveTo(err_left, y)
        path.lineTo(err_left, y-offset)
        path.lineTo(err_left, y+offset)
        path.moveTo(err_left, y)
        path.lineTo(err_right, y)
        path.lineTo(err_right, y-offset)
        path.lineTo(err_right, y+offset)
        qp.drawPath(path)

