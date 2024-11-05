import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode
import numpy as np
import time

# Remove the following import
# from surface_distance import metrics

# Add import for lunginator_3000 and compute_dice_coefficient functions
from algorithm import lunginator_3000, compute_dice_coefficient


#
# Lungs_script
#


class Lungs_script(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Lungs_script")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Lungs_script">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Lungs_script1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Lungs_script",
        sampleName="Lungs_script1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "Lungs_script1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="Lungs_script1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="Lungs_script1",
    )

    # Lungs_script2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Lungs_script",
        sampleName="Lungs_script2",
        thumbnailFileName=os.path.join(iconsPath, "Lungs_script2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="Lungs_script2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="Lungs_script2",
    )


#
# Lungs_scriptParameterNode
#


@parameterNodeWrapper
class Lungs_scriptParameterNode:
    """
    The parameters needed by the module.

    inputVolume - The input 3D image volume.
    totalSegmentation - The segmentation from Total Segmentator.
    groundTruth - The ground truth volume.
    outputVolume - The output segmentation node (custom segmentation).
    """

    inputVolume: vtkMRMLScalarVolumeNode
    totalSegmentation: vtkMRMLSegmentationNode
    groundTruth: vtkMRMLScalarVolumeNode
    outputVolume: vtkMRMLSegmentationNode


#
# Lungs_scriptWidget
#


class Lungs_scriptWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Lungs_script.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = Lungs_scriptLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Remove the first connection to onApplyButton to eliminate duplication
        # self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Connect the "Apply" button to the single onApplyButton method
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[Lungs_scriptParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:        
        if self._parameterNode and \
           self._parameterNode.inputVolume and \
           self._parameterNode.totalSegmentation and \
           self._parameterNode.groundTruth and \
           self._parameterNode.outputVolume:
            self.ui.applyButton.toolTip = _("Ready to compute results")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select all input and output nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks 'Apply' button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Run processing
            self.logic.process(
                self._parameterNode.inputVolume,
                self._parameterNode.outputVolume,
                self._parameterNode.totalSegmentation,
                self._parameterNode.groundTruth)


#
# Lungs_scriptLogic
#


class Lungs_scriptLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return Lungs_scriptParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputSegmentation: vtkMRMLSegmentationNode,
                totalSegmentation: vtkMRMLSegmentationNode,
                groundTruth: vtkMRMLScalarVolumeNode) -> None:
        """
        Run the processing algorithm.
        :param inputVolume: 3D image volume
        :param outputSegmentation: output segmentation node (custom segmentation)
        :param totalSegmentation: segmentation from total segmentator
        :param groundTruth: ground truth volume
        """
        if not inputVolume or not outputSegmentation or not totalSegmentation or not groundTruth:
            raise ValueError("All input parameters must be set.")

        startTime = time.time()
        logging.info('Processing started')

        # Ensure output segmentation is empty
        outputSegmentation.GetSegmentation().RemoveAllSegments()

        # Get NumPy arrays from volumes
        input_array = slicer.util.arrayFromVolume(inputVolume)
        ground_truth_array = slicer.util.arrayFromVolume(groundTruth)

        # Use lunginator_3000 to perform custom lung segmentation
        segmentation_result = lunginator_3000(input_array)

        # Extract left and right lung masks from the result
        left_lung_mask = (segmentation_result == 1)
        right_lung_mask = (segmentation_result == 2)

        # Create segmentation from the result
        outputSegmentation.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        # Add segments for left and right lungs
        segmentIDL = outputSegmentation.GetSegmentation().AddEmptySegment("Custom Left Lung")
        segmentIDR = outputSegmentation.GetSegmentation().AddEmptySegment("Custom Right Lung")

        # Update segments with the masks
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            left_lung_mask.astype(np.uint8), outputSegmentation, segmentIDL, inputVolume)
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            right_lung_mask.astype(np.uint8), outputSegmentation, segmentIDR, inputVolume)

        # Get ground truth masks for left and right lungs
        gt_left_lung = (ground_truth_array == 2)
        gt_right_lung = (ground_truth_array == 3)

        # Compute dice coefficients between custom segmentation and ground truth
        dice_coef_left = compute_dice_coefficient(
            left_lung_mask.astype(bool), gt_left_lung.astype(bool))
        dice_coef_right = compute_dice_coefficient(
            right_lung_mask.astype(bool), gt_right_lung.astype(bool))

        # Print dice coefficients for custom segmentation vs ground truth
        print(f"Dice coefficient (Custom vs Ground Truth) for left lung: {dice_coef_left}")
        print(f"Dice coefficient (Custom vs Ground Truth) for right lung: {dice_coef_right}")

        # Combine left lung segments from total segmentation
        segment_ids = totalSegmentation.GetSegmentation().GetSegmentIDs()
        left_lung_ids = ['lung_upper_lobe_left', 'lung_middle_lobe_left', 'lung_lower_lobe_left']
        right_lung_ids = ['lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right']

        total_left_lung_mask = None
        for lung_id in left_lung_ids:
            if lung_id in segment_ids:
                l_mask = slicer.util.arrayFromSegment(totalSegmentation, lung_id)
                if total_left_lung_mask is None:
                    total_left_lung_mask = l_mask.copy()
                else:
                    total_left_lung_mask |= l_mask

        total_right_lung_mask = None
        for lung_id in right_lung_ids:
            if lung_id in segment_ids:
                r_mask = slicer.util.arrayFromSegment(totalSegmentation, lung_id)
                if total_right_lung_mask is None:
                    total_right_lung_mask = r_mask.copy()
                else:
                    total_right_lung_mask |= r_mask

        # Compute dice coefficients between custom segmentation and total segmentation
        dice_coef_total_left = compute_dice_coefficient(
            left_lung_mask.astype(bool), total_left_lung_mask.astype(bool))
        dice_coef_total_right = compute_dice_coefficient(
            right_lung_mask.astype(bool), total_right_lung_mask.astype(bool))

        # Print dice coefficients for custom segmentation vs total segmentation
        print(f"Dice coefficient (Custom vs Total Segmentation) for left lung: {dice_coef_total_left}")
        print(f"Dice coefficient (Custom vs Total Segmentation) for right lung: {dice_coef_total_right}")

        logging.info(f'Processing completed in {time.time() - startTime:.2f} seconds')