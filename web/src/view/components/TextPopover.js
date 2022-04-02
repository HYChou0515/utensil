import {
  EuiButton,
  EuiButtonIcon,
  EuiFieldText,
  EuiPopover,
  EuiPopoverFooter,
} from "@elastic/eui";
import React, { useState } from "react";

const TextPopover = ({ iconType, display, placeholder, onSubmit }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [textValue, setTextValue] = useState();
  const onClick = () => {
    onClose();
    onSubmit(textValue);
  };
  const onClose = () => {
    setIsOpen(false);
    setTextValue();
  };
  return (
    <EuiPopover
      button={
        <EuiButtonIcon
          display={display}
          iconType={iconType}
          onClick={() => setIsOpen(!isOpen)}
        />
      }
      isOpen={isOpen}
      closePopover={onClose}
    >
      <EuiFieldText
        placeholder={placeholder}
        value={textValue}
        onChange={(e) => setTextValue(e.target.value)}
      />
      <EuiPopoverFooter>
        <EuiButton fullWidth size="s" onClick={onClick}>
          Confirm
        </EuiButton>
      </EuiPopoverFooter>
    </EuiPopover>
  );
};

export default TextPopover;
