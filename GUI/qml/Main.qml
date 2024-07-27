import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts
import QtQuick.Controls.Material
import QtCore
import Qt.labs.platform
import io.qt.textproperties 1.0

Window {
    id:window
    title: "Wizard"
    width: 640
    height: 550
    minimumWidth:  640
    minimumHeight: 500
    visible: true
    Material.theme: Material.Dark 
    Bridge {
        id: bridge
    }
    Settings {
        id: config
        location: Qt.resolvedUrl("config.ini")
        category: "config"
        property string force_field_path:"" 
        property string symbol_formula:""
        property int symbol_structure:0
        property string symbol_lattice_a:""
        property string symbol_lattice_c:"0"
        property string result: ""
        property int result_source: 0
    }
    Loader {
        id: active_page
        source: "StartPage.qml"
        anchors.fill: parent
        Connections {
            target: active_page.item
            function onChangeActivePage(name) {
                active_page.source = name
            }
        }
    }
}


