import styled from "@emotion/styled";
import React, { useState } from "react";
import { useSelector } from "react-redux";

import apiClient from "../../api/api";

const GalleryItemContent = styled.div`
  color: black;
  font-family: Helvetica, Arial, serif;
  padding: 5px;
  border: solid 1px black;
  border-radius: 5px;
  margin: 0px 10px 2px;
  cursor: pointer;
  background: ${(p) => p.color};
`;

const GalleryItemBox = styled.div`
  min-width: 172px;
  flex-grow: 0;
  flex-shrink: 0;
`;

const GalleryItemWidget = ({ model, name }) => {
  return (
    <GalleryItemBox>
      <GalleryItemContent
        color={model.color}
        draggable
        onDragStart={(event) => {
          event.dataTransfer.setData(
            "storm-diagram-node",
            JSON.stringify(model)
          );
        }}
      >
        {name}
      </GalleryItemContent>
    </GalleryItemBox>
  );
};

export default GalleryItemWidget;
