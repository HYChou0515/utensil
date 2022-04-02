import { EuiFlexGroup, EuiFlexItem, EuiPanel } from "@elastic/eui";
import React from "react";

import GalleryItemWidget from "../../components/GalleryItemWidget";

const NodeGallery = () => {
  return (
    <EuiPanel>
      <EuiFlexGroup direction="column" alignitems="center">
        <EuiFlexItem>
          <GalleryItemWidget
            model={{ type: "in", color: "rgb(192,255,0)" }}
            name="Input Node"
          />
        </EuiFlexItem>
        <EuiFlexItem>
          <GalleryItemWidget
            model={{ type: "out", color: "rgb(0,192,255)" }}
            name="Output Node"
          />
        </EuiFlexItem>
      </EuiFlexGroup>
    </EuiPanel>
  );
};

export default NodeGallery;
