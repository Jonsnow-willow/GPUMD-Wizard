import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts
import QtQuick.Controls.Material
import QtCore
import Qt.labs.platform
import io.qt.textproperties 1.0
GroupBox {
    title: qsTr("Crystal structure")
    ColumnLayout {
        width: parent.width
        spacing: 10
        RowLayout
        {
            Layout.fillWidth: true
            TextField {
                id: formula
                Layout.fillWidth: true
                text: config.symbol_formula
                placeholderText: qsTr("Formula")
                onTextChanged: {
                    config.symbol_formula = text
                }
            }
            ComboBox {
                id: lattice_structure
                textRole: "key"
                model: ListModel {
                    ListElement { key: "FCC"; value: 123 }
                    ListElement { key: "BCC"; value: 456 }
                    ListElement { key: "HCP"; value: 789 }
                }
                currentIndex: config.symbol_structure 
                onCurrentIndexChanged: {
                    config.symbol_structure = currentIndex
                }
            }
            TextField {
                id: lattice_a
                width: 120
                placeholderText: qsTr("Lattice a")
                text: config.symbol_lattice_a
                onTextChanged: {
                    config.symbol_lattice_a = text
                }
            }
        }
        RowLayout
        {
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignRight
            TextField {
                id: lattice_c
                visible: lattice_structure.currentIndex === 2
                width: 120
                placeholderText: qsTr("Lattice c")
                text: config.symbol_lattice_c
                onTextChanged: {
                    config.symbol_lattice_c = text
                }
            }
        }
    }
}