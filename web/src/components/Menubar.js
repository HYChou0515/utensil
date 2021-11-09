import React, {Component} from "react";
import {Menu, Icon} from "semantic-ui-react";

class Menubar extends Component {
    state = {}

    handleItemClick = (e, { name }) => this.setState({ activeItem: name })

    render() {
        const { activeItem } = this.state

        return (
            <Menu icon>
                <Menu.Item
                    name='file'
                    active={activeItem === 'file'}
                    onClick={this.handleItemClick}
                >
                    <Icon name='file' />
                </Menu.Item>

                <Menu.Item
                    name='folder open'
                    active={activeItem === 'folder open'}
                    onClick={this.handleItemClick}
                >
                    <Icon name='folder open' />
                </Menu.Item>

                <Menu.Item
                    name='save'
                    active={activeItem === 'save'}
                    onClick={this.handleItemClick}
                >
                    <Icon name='save' />
                </Menu.Item>

                <Menu.Item
                    name='clone'
                    active={activeItem === 'clone'}
                    onClick={this.handleItemClick}
                >
                    <Icon name='clone' />
                </Menu.Item>
            </Menu>
        )
    }
};

export default Menubar;
