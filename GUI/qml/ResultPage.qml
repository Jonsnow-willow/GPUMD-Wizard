import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts
import QtQuick.Controls.Material
import QtCore
import Qt.labs.platform
import io.qt.textproperties 1.0
Page{
    signal changeActivePage(variant name)
    id:page_result
    visible: true
    anchors.fill: parent
    Flickable {
        anchors.fill: parent
        contentHeight: item_0.height +
                        // item_1.height +
                        // item_2.height +
                        4*g_spacing +
                        return_button.height
        height: contentHeight
        ScrollBar.vertical: ScrollBar{
        }
        GroupBox {
            id: item_0
            title: qsTr("Phonon dispersion")
            anchors{
                top: parent.top
                left: parent.left
                right: parent.right
                margins: g_spacing
            }
            height: 400
            Image {
                anchors.fill: parent
                source: config.result
                fillMode: Image.PreserveAspectFit
            }
        }
        
        Button {
            id: return_button
            anchors{
                right: parent.right
                top: item_0.bottom
                margins: g_spacing
            }
            text: qsTr("Return")
            onClicked: {
                if (config.result_source==0)
                    changeActivePage("StartPage.qml")
                if (config.result_source==1)
                    changeActivePage("HistoricalPage.qml")
            }
        }
    }
}