import styled from "@emotion/styled";
import {
  AiFillPlusCircle,
  AiOutlineLogin,
  AiOutlineLogout,
} from "react-icons/ai";

export const IconStack = styled.span`
  display: grid;
  svg {
    grid-area: 1 / 1;
  }
`;

export const InPortIcon = () => (
  <IconStack>
    <AiOutlineLogin style={{ transform: "translate(-15%, -15%)" }} />
    <AiFillPlusCircle
      style={{ transform: "translate(30%, 30%) scale(0.8,0.8)" }}
    />
  </IconStack>
);

export const OutPortIcon = () => (
  <IconStack>
    <AiOutlineLogout style={{ transform: "translate(-15%, -15%)" }} />
    <AiFillPlusCircle
      style={{ transform: "translate(30%, 30%) scale(0.8,0.8)" }}
    />
  </IconStack>
);
