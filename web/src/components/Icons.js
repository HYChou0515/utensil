import styled from "@emotion/styled";
import {
  AiFillMinusCircle,
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

export const AddInPortIcon = () => (
  <IconStack>
    <AiOutlineLogin style={{ transform: "translate(-15%, -15%)" }} />
    <AiFillPlusCircle
      style={{ transform: "translate(30%, 30%) scale(0.8,0.8)" }}
    />
  </IconStack>
);

export const AddOutPortIcon = () => (
  <IconStack>
    <AiOutlineLogout style={{ transform: "translate(-15%, -15%)" }} />
    <AiFillPlusCircle
      style={{ transform: "translate(30%, 30%) scale(0.8,0.8)" }}
    />
  </IconStack>
);

export const DeleteInPortIcon = () => (
  <IconStack>
    <AiOutlineLogin style={{ transform: "translate(-15%, -15%)" }} />
    <AiFillMinusCircle
      style={{ transform: "translate(30%, 30%) scale(0.8,0.8)" }}
    />
  </IconStack>
);

export const DeleteOutPortIcon = () => (
  <IconStack>
    <AiOutlineLogout style={{ transform: "translate(-15%, -15%)" }} />
    <AiFillMinusCircle
      style={{ transform: "translate(30%, 30%) scale(0.8,0.8)" }}
    />
  </IconStack>
);
