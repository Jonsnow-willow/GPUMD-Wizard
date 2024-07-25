
import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts
import QtQuick.Controls.Material
import QtCore
import Qt.labs.platform
import io.qt.textproperties 1.0
Page{
    id: page_historical
    signal changeActivePage(variant name)
    visible: true
    anchors.fill: parent
    Flickable {
        anchors.fill: parent
        contentHeight: 400
        height: contentHeight
        ScrollBar.vertical: ScrollBar{
        }
        ListView {
            anchors.fill: parent
            id: list_view
            model: listModel
            delegate: ItemDelegate {
                text: model.name
                width: parent.width
                onClicked: {
                    config.result_source = 1
                    config.result = model.data
                    changeActivePage("ResultPage.qml")
                }
            }
        }
        Button {
            id: return_button2
            anchors{
                right: parent.right
                top: list_view.bottom
                margins: g_spacing
            }
            text: qsTr("Return")
            onClicked: {
                changeActivePage("StartPage.qml")
            }
        }
    }
    ListModel {
        id: listModel
    }
    Component.onCompleted: {
        var data = bridge.get_historical()
        console.log("data.length:", data.length);
        for (var i = 0; i < data.length; i++) {
            listModel.append(data[i]);
        }
    }
}