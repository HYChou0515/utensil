import {
  EuiHeader,
  EuiHeaderSection,
  EuiHeaderSectionItem,
  EuiHeaderSectionItemButton,
  EuiIcon,
} from "@elastic/eui";
import React from "react";
import { BsGearFill } from "react-icons/bs";
import { FaCubes, FaFolderOpen, FaSitemap } from "react-icons/fa";
import { useDispatch, useSelector } from "react-redux";

import apiClient from "../../../api/api";
import canvasDomain from "../../../domain/CanvasDomain";
import logo from "../../../logo.svg";
import {
  toggleShowGallery,
  toggleShowOpenFileUi,
  toggleShowSettingUi,
  toggleUsedLayout,
} from "../../../store/features/canvas/flowEditor";

const Menu = () => {
  const usedLayout = useSelector((state) => state.flowEditor.usedLayout);
  const dispatch = useDispatch();
  return (
    <EuiHeader>
      <EuiHeaderSection grow={false}>
        <EuiHeaderSectionItem>
          <EuiHeaderSectionItemButton
            onClick={() => {
              console.log(canvasDomain.diagramEngine.getModel().serialize());
              apiClient
                .postGraph(canvasDomain.diagramEngine.getModel().serialize())
                .then((a) => console.log(a));
            }}
          >
            <EuiIcon type={logo} size="l" />
          </EuiHeaderSectionItemButton>
        </EuiHeaderSectionItem>
        <EuiHeaderSectionItem>
          <EuiHeaderSectionItemButton
            onClick={() => dispatch(toggleShowOpenFileUi())}
          >
            <EuiIcon type={FaFolderOpen} />
          </EuiHeaderSectionItemButton>
        </EuiHeaderSectionItem>
      </EuiHeaderSection>

      <EuiHeaderSection side="right">
        <EuiHeaderSectionItem>
          <EuiHeaderSectionItemButton
            onClick={() => dispatch(toggleShowSettingUi())}
          >
            <EuiIcon type={BsGearFill} />
          </EuiHeaderSectionItemButton>
        </EuiHeaderSectionItem>
        <EuiHeaderSectionItem>
          <EuiHeaderSectionItemButton
            onClick={() => dispatch(toggleShowGallery())}
          >
            <EuiIcon type={FaCubes} />
          </EuiHeaderSectionItemButton>
        </EuiHeaderSectionItem>
        <EuiHeaderSectionItem>
          <EuiHeaderSectionItemButton
            onClick={() => dispatch(toggleUsedLayout())}
          >
            <EuiIcon
              type={FaSitemap}
              style={{
                transform: `rotate(${usedLayout === "TB" ? 0 : 270}deg)`,
              }}
            />
          </EuiHeaderSectionItemButton>
        </EuiHeaderSectionItem>
      </EuiHeaderSection>
    </EuiHeader>
  );
};

export default Menu;
