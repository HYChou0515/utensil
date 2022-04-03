import { EuiDataGrid } from "@elastic/eui";
import React, { useCallback, useState } from "react";

const DataGrid = ({
  data,
  columns,
  initColumnVis,
  initSorting,
  initPageSizeOptions,
}) => {
  const [columnVis, setColumnVis] = useState(
    initColumnVis ?? columns.map((c) => c.id)
  );
  const [sorting, setSorting] = useState(initSorting ?? []);
  const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: 10 });
  const pageSizeOptions = initPageSizeOptions ?? [10, 20, 30];
  const onChangeItemsPerPage = useCallback(
    (pageSize) =>
      setPagination((pagination) => ({
        ...pagination,
        pageSize,
        pageIndex: 0,
      })),
    [setPagination]
  );
  const onChangePage = useCallback(
    (pageIndex) =>
      setPagination((pagination) => ({ ...pagination, pageIndex })),
    [setPagination]
  );
  const renderCellValue = useCallback(
    ({ rowIndex, columnId }) => data[rowIndex][columnId],
    [data]
  );
  return (
    <EuiDataGrid
      rowCount={data.length}
      columns={columns}
      columnVisibility={{
        visibleColumns: columnVis,
        setVisibleColumns: setColumnVis,
      }}
      renderCellValue={renderCellValue}
      inMemory={{ level: "sorting" }}
      sorting={{
        columns: sorting,
        onSort: setSorting,
      }}
      pagination={{
        ...pagination,
        pageSizeOptions: pageSizeOptions,
        onChangePage: onChangePage,
        onChangeItemsPerPage: onChangeItemsPerPage,
      }}
      gridStyle={{
        border: "all",
        fontSize: "m",
        cellPadding: "m",
        stripes: true,
        rowHover: "highlight",
        header: "shade",
      }}
      rowHeightsOptions={{
        defaultHeight: 34,
        rowHeights: {
          0: "auto",
        },
        lineHeight: "1em",
      }}
    />
  );
};

export default DataGrid;
