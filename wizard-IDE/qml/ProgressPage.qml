import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts
import QtQuick.Controls.Material
import QtCore
import Qt.labs.platform
import io.qt.textproperties 1.0

Page {
    signal changeActivePage(variant name)
    id: page_progress
    visible: true
    anchors.fill: parent
    Label {
        id: calculating
        anchors{
            bottom: progress_bar.top
            margins: g_spacing
            horizontalCenter: parent.horizontalCenter
        }
        text: qsTr("calculating...")
    }
    ProgressBar {
        id: progress_bar
        anchors.centerIn: parent
        value: 0.0
    }
    
    Timer {
        id: timer
        interval: 10
        repeat: true 
        onTriggered: {
            progress_bar.value = progress_bar.value + 0.01
            var result = bridge.get_result()
            if(result)
            {
                progress_bar.value = 1.0
                config.result = result
                config.result_source = 0
                changeActivePage("ResultPage.qml")
                stop()
            }
            if(progress_bar.value<1.0)
                return
            progress_bar.value = 0.99
        }
    }
    Component.onCompleted: {
        timer.start()
        bridge.calculate_properties([config.force_field_path, config.symbol_formula, config.symbol_structure, config.symbol_lattice_a, config.symbol_lattice_c])
    }
}