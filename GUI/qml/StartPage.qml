import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts
import QtQuick.Controls.Material
import QtCore
import Qt.labs.platform
import io.qt.textproperties 1.0
Page {
    signal changeActivePage(variant name)
    id: page_start
    property var selectedItems: []

    anchors.fill: parent
    Flickable {
        anchors.fill: parent
        contentHeight: column_layout.height + 40
        ScrollBar.vertical: ScrollBar{
        }
        Item {
            anchors.fill: parent
            anchors.margins: 20
            ColumnLayout {
                id: column_layout
                width: parent.width
                spacing: 10
                RowLayout {
                    Layout.fillWidth: true
                    anchors.topMargin: 10
                    spacing: 10
                    height: childrenRect.height
                    Label {
                        id: title_label
                        text: qsTr("Wizard")
                        font.pointSize: 30
                    } 
                    Item{
                        Layout.fillWidth: true
                    }
                    Button {
                        id: theme
                        onClicked: {
                            window.Material.theme = window.Material.theme == Material.Dark ? Material.Light : Material.Dark
                            text = window.Material.theme == Material.Dark ? "Light" : "Dark" 
                        }
                        text: qsTr("Light")
                    }
                    Button {
                        id: results_list
                        onClicked: {
                            changeActivePage("HistoricalPage.qml")
                        }
                        text: qsTr("Historical results")
                    }
                }
                GroupBox {
                    id: force_field
                    Layout.fillWidth: true
                    title: qsTr("Force Field")
                    ColumnLayout {
                        width: parent.width
                        spacing: 10
                        RowLayout {
                            Layout.fillWidth: true
                            TextField {
                                id: force_text_field
                                text: config.force_field_path
                                Layout.fillWidth: true
                                placeholderText: qsTr("Force field")
                                onTextChanged: {
                                    // repeater.model = bridge.read_force_field(text)
                                }
                            }
                            Button {
                                text: qsTr("Select force field")
                                onClicked:  {
                                    file_dialog.file = "file:///" + force_text_field.text
                                    file_dialog.open()
                                }
                            }
                        }
                        TextArea {
                            id: force_field_infor
                            readOnly: true
                            Layout.fillWidth: true
                            text: {
                                var result = ""
                                var temp = bridge.get_force_field(force_text_field.text)
                                result = result + temp[2] +  "<br>"
                                return result.slice(0, -4)
                            }
                            textFormat: Text.RichText
                        }
                    }
                }
                GroupBox{
                    id: stack_layout
                    title: qsTr("Model")
                    Layout.fillWidth: true
                    spacing: 10
                    ColumnLayout {
                        width: parent.width
                        spacing: 10
                        TabBar {
                            id: tab_bar
                            Layout.fillWidth: true
                            TabButton {
                                Layout.fillWidth: true
                                text: qsTr("Calculation properties")
                            }
                            TabButton {
                                Layout.fillWidth: true
                                text: qsTr("Generate train set")
                            }
                            TabButton {
                                Layout.fillWidth: true
                                text: qsTr("GPUMD IO")
                            }
                        }
                        StackLayout {
                            Layout.fillWidth: true
                            currentIndex: tab_bar.currentIndex
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                CrystalStructure{
                                    Layout.fillWidth: true
                                }
                                GroupBox {
                                    title: qsTr("Calculation Items")
                                    Layout.fillWidth: true
                                    Flow{
                                        width: parent.width
                                        spacing: 10
                                        CheckDelegate {
                                            text: qsTr("Lattice constant")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Elastic constant")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Eos curve")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Phonon dispersion")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Formation energy vacancy")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Migration energy vacancy")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Formation energy divacancies")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Formation energy sia")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Formation energy surface")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Stacking fault")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Bcc metal screw dipole move")
                                        }
                                        CheckDelegate {
                                            text: qsTr("Bcc metal screw one move")
                                        }
                                    }
                                }
                            }
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                CrystalStructure{
                                    Layout.fillWidth: true
                                }
                                GroupBox {
                                    title: qsTr("Calculation Items")
                                    Layout.fillWidth: true
                                    RowLayout{
                                        Layout.fillWidth: true
                                        RadioButton {
                                            text: qsTr("mono-vacancy")
                                        }
                                        RadioButton {
                                            text: qsTr("di-vacancies")
                                        }
                                        RadioButton {
                                            text: qsTr("self-interstitial")
                                        }
                                    }
                                }
                                GroupBox {
                                    title: qsTr("Molecular Dynamics")
                                    Layout.fillWidth: true
                                    RowLayout
                                    {
                                        Layout.fillWidth: true
                                        spacing: 10
                                        ComboBox {
                                            id:  ensemble
                                            width: 100
                                            textRole: "key"
                                            model: ListModel {
                                                ListElement { key: "NVE"; value: 123 }
                                                ListElement { key: "NVT"; value: 456 }
                                                ListElement { key: "NPT"; value: 789 }
                                                ListElement { key: "NPH"; value: 789 }
                                                ListElement { key: "μVT"; value: 789 }
                                            }
                                            currentIndex: 0
                                            onCurrentIndexChanged: {
                                            }
                                        }
                                        TextField {
                                            id: temperature
                                            text: "0"
                                            placeholderText: qsTr("Temperature")
                                        }
                                        TextField {
                                            id: steps
                                            text: "1"
                                            placeholderText: qsTr("Steps")
                                        }
                                        TextField {
                                            visible: ensemble.currentIndex === 2
                                            id: pressure
                                            text: "0"
                                            placeholderText: qsTr("Pressure")
                                        }
                                    }
                                }
                            }
                            Item {
                                Layout.fillWidth: true
                                height: 0
                            }
                        }
                    }
                }
                RowLayout
                {
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignRight
                    Button {
                        id: start_button
                        text: qsTr("Start!")
                        onClicked: {
                            changeActivePage("ProgressPage.qml")
                        }
                    }
                }
            }
        }
    }
    FileDialog {
        id: file_dialog
        onAccepted:{
            config.force_field_path = '/' + file.toString().substr(8);
            bridge.read_force_field(force_text_field.text)
            force_field_infor.text = bridge.get_force_field(force_text_field.text)
        }
        nameFilters: ["Nep (*.txt)"]
    }
}